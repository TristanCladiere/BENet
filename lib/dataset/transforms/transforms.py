from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
from torchvision.transforms import ColorJitter, GaussianBlur


def intersection(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    x, y, x2, y2 = bb1
    bb1_area = (x2 - x) * (y2 - y)

    if x_right < x_left or y_bottom < y_top:
        inter = 0.0
    else:
        inter = (x_right - x_left) * (y_bottom - y_top) / bb1_area

    return inter


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, image_type, bbox):
        for t in self.transforms:
            image, image_type, bbox = t(image, image_type, bbox)
        return image, bbox

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomMaskSubject(object):
    def __init__(self):
        self.thr = 0.5

    def __call__(self, image, anno, all_anns):
        num_people = len(all_anns)

        if num_people > 1:
            indexes = np.random.choice(num_people, num_people, replace=False)
            new_anno = [all_anns[indexes[0]]]
            vis_ratio = [1]

            # 1: to be sure to keep at least 1 annotation
            for ind in indexes[1:]:
                if np.random.rand() > self.thr:
                    x, y, x2, y2 = all_anns[ind]["bbox"]

                    ok = True
                    i = 0
                    # Check if this mask can be applied without overlapping too much with the other bboxes
                    while ok and i < len(new_anno):
                        xs, ys, x2s, y2s = new_anno[i]["bbox"]

                        # Check if this mask hide the centers of the previous ones, or has its center hidden
                        if x <= (xs+x2s)/2 <= x2 or y <= (ys+y2s)/2 <= y2 or \
                                xs <= (x+x2)/2 <= x2s or ys <= (y+y2)/2 <= y2s:
                            ok = False
                        else:
                            # Also check if the preserved bbox areas are not hidden too much by the mask
                            vis_ratio[i] -= intersection(new_anno[i]["bbox"], all_anns[ind]["bbox"])
                            if vis_ratio[i] < 0.6:
                                ok = False
                        i += 1

                    if ok:
                        image[y:y2, x:x2, :] = 0
                    else:
                        new_anno.append(all_anns[ind])
                        vis_ratio.append(1)
                else:
                    new_anno.append(all_anns[ind])
                    vis_ratio.append(1)
        else:
            new_anno = all_anns

        return [image], new_anno, all_anns


class MaskAllSubjects(object):
    def __call__(self, image, anno, all_anns):
        for obj in all_anns:
            x, y, x2, y2 = obj['bbox']
            image[y:y2, x:x2, :] = 0

        return [image], anno, all_anns


class ExtractSubject(object):
    def __call__(self, image, anno, all_anns):
        x, y, x2, y2 = all_anns[0]['bbox']
        image = image[y:y2, x:x2, :]

        return [image], anno, all_anns


class Fusion(object):
    def __init__(self):
        self.transforms = [ExtractSubject(), MaskAllSubjects()]

    def __call__(self, image, anno, all_anns):
        if type(image) is not np.ndarray:
            image = np.copy(image)
        copied = [image]
        for t in self.transforms:
            temp, anno, all_anns = t(np.copy(image), anno, all_anns)
            copied.append(temp[0])

        return copied, anno, all_anns


class RandomChoice(object):
    def __init__(self, transforms, p):
        self.transforms = transforms
        self.p = p
        assert len(transforms) == len(self.p), "We don't have the same number of transformations and probabilities"

    def __call__(self, image, anno, all_anns):
        copied = np.copy(image)
        height, width = copied.shape[:2]
        img_area = height * width

        # Check if the image can be used to train the pc head (main bbox does not overlap too much with other bboxes),
        # and if it can be used to train the context head (i.e. it has enough background information)
        bb_tot_area = 0
        tot_inter = 0

        for i, obj in enumerate(all_anns):
            x, y, x2, y2 = obj['bbox']
            bb_area = (x2 - x) * (y2 - y)
            bb_tot_area += bb_area

            if i > 0:
                tot_inter += intersection(all_anns[0]["bbox"], obj["bbox"])

        if bb_tot_area > 0.4 * img_area or tot_inter > 0.4:
            transforms = []
            proba = []
            recycled_proba = []
            for i, t in enumerate(self.transforms):
                class_name = t.__class__.__name__
                if class_name == "MaskAllSubjects" and bb_tot_area > 0.4 * img_area:
                    recycled_proba.append(self.p[i])
                elif class_name == "ExtractSubject" and tot_inter > 0.4:
                    recycled_proba.append(self.p[i])
                else:
                    transforms.append(t)
                    proba.append(self.p[i])
            for rp in recycled_proba:
                rp = rp / len(proba)
                proba = [p+rp for p in proba]
        else:
            transforms = self.transforms
            proba = self.p

        t = random.choices(transforms, weights=proba)[0]
        class_name = t.__class__.__name__
        if class_name == "str":
            head = "det"
            copied = [copied]
        else:
            copied, anno, all_anns = t(copied, anno, all_anns)
            if class_name == "RandomMaskSubject":
                head = "bu"
            elif class_name == "MaskAllSubjects":
                head = "context"
            elif class_name == "ExtractSubject":
                head = "pc"
            elif class_name == "Fusion":
                head = "fusion"

        return {"image": copied, "head": head}, anno, all_anns


class ColorJitterPerso(object):
    def __init__(self):
        self.color_jitter = ColorJitter(brightness=(0.85, 1), contrast=(0.85, 1), saturation=(0.85, 1), hue=(0.0, 0.10))

    def __call__(self, image, image_type, bbox):
        _, b, c, s, h = self.color_jitter.get_params((0.85, 1), (0.85, 1), (0.85, 1), (0.0, 0.10))
        for i, img in enumerate(image):
            image[i][0:3] = F.adjust_brightness(img[0:3], b)
            image[i][0:3] = F.adjust_contrast(img[0:3], c)
            image[i][0:3] = F.adjust_saturation(img[0:3], s)
            image[i][0:3] = F.adjust_hue(img[0:3], h)

        return image, image_type, bbox


class RandomGaussianNoise(object):
    def __init__(self, mean=0.0, min_var=0.0, max_var=0.0005):
        self.mean = mean
        self.min_var = min_var
        self.max_var = max_var

    def __call__(self, image, image_type, bbox):
        var = torch.rand(1)
        std = ((self.max_var - self.min_var)*var + self.min_var)**0.5
        dim = image[0][0:3].size() if isinstance(image, list) else image[0:3].size()
        noise = torch.normal(self.mean, std.item(), size=dim)

        for i, img in enumerate(image):
            image[i][0:3] = torch.clip(img[0:3]+noise, 0, 1)

        return image, image_type, bbox


class GaussianBlurPerso(object):
    def __init__(self):
        self.gaussian_blur = GaussianBlur(3)

    def __call__(self, image, image_type, bbox):
        sigma = self.gaussian_blur.get_params(0.1, 2.0)  # default sigma values interval
        for i, img in enumerate(image):
            image[i][0:3] = F.gaussian_blur(img[0:3], 3, sigma)

        return image, image_type, bbox


class ToTensor(object):
    def __call__(self, image, image_type, bbox):
        for i, img in enumerate(image):
            image[i] = F.to_tensor(img)

        return image, image_type, bbox


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, image_type, bbox):
        for i, img in enumerate(image):
            image[i] = F.normalize(img, mean=self.mean[:img.shape[0]], std=self.std[:img.shape[0]])

        return image, image_type, bbox


class RandomHorizontalFlip(object):
    def __init__(self, output_size, prob=0.5):
        self.prob = prob
        self.output_size = output_size if isinstance(output_size, list) \
            else [output_size]

    def __call__(self, image, image_type, bbox):
        assert isinstance(bbox, list)
        if random.random() < self.prob:
            for i, img in enumerate(image):
                image[i] = img[:, ::-1] - np.zeros_like(img)

            if bbox:
                for i, _output_size in enumerate(self.output_size):
                    bbox[i][:, [0, 2]] = _output_size - bbox[i][:, [2, 0]] - 1

        return image, image_type, bbox


class RandomAffineTransform(object):
    def __init__(self,
                 input_size,
                 output_size,
                 max_rotation,
                 min_scale,
                 max_scale,
                 scale_type,
                 max_translate,
                 perspective=True):
        self.input_size = input_size
        self.output_size = output_size if isinstance(output_size, list) \
            else [output_size]

        self.max_rotation = max_rotation
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_type = scale_type
        self.max_translate = max_translate
        self.perspective = perspective

    def _get_affine_matrix(self, center, scale, res, rot=0, perspective=True):
        # Generate transformation matrix
        h = 200 * scale
        t = np.zeros((3, 3))
        t[0, 0] = float(res[1]) / h
        t[1, 1] = float(res[0]) / h
        t[0, 2] = res[1] * (-float(center[0]) / h + .5)
        t[1, 2] = res[0] * (-float(center[1]) / h + .5)
        t[2, 2] = 1
        if not rot == 0:
            rot = -rot  # To match direction of rotation from cropping
            rot_mat = np.zeros((3, 3))
            rot_rad = rot * np.pi / 180
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0, :2] = [cs, -sn]
            rot_mat[1, :2] = [sn, cs]
            rot_mat[2, 2] = 1
            # Need to rotate around center
            t_mat = np.eye(3)
            t_mat[0, 2] = -res[1]/2
            t_mat[1, 2] = -res[0]/2
            t_inv = t_mat.copy()
            t_inv[:2, 2] *= -1
            t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
            if perspective:
                t[2, 0] = (np.random.random(1) - 0.5) * 1e-4
                t[2, 1] = (np.random.random(1) - 0.5) * 1e-4
        return t

    def _perspective_transform(self, bbox, mat):
        bbox = np.dot(np.concatenate((bbox, bbox[:, 0:1]*0+1), axis=1), mat.T)
        for i in range(bbox.shape[0]):
            bbox[i] = bbox[i] / bbox[i, 2]

        return bbox[:, :2]

    def __call__(self, images, image_type, bbox):
        assert isinstance(images, list)
        assert isinstance(bbox, list)
        aug_scale = np.random.random() * (self.max_scale - self.min_scale) + self.min_scale
        aug_rot = (np.random.random() * 2 - 1) * self.max_rotation
        dx = np.random.random()
        dy = np.random.random()
        for i, image in enumerate(images):
            height, width = image.shape[:2]
            center = np.array((width/2, height/2))
            if self.scale_type == 'long' or image_type == "pc" or (image_type == "fusion" and i == 1):
                scale = max(height, width)/200
            elif self.scale_type == 'short':
                scale = min(height, width)/200
            else:
                raise ValueError('Unknown scale type: {}'.format(self.scale_type))

            scale *= aug_scale

            if self.max_translate > 0:
                _dx = scale * self.max_scale * dx
                _dy = scale * self.max_scale * dy
                center[0] += _dx
                center[1] += _dy

            for j, _output_size in enumerate(self.output_size):
                mat_output = self._get_affine_matrix(
                    center, scale, (_output_size, _output_size), aug_rot
                )
                if bbox and i == 0:
                    bbox[j][:, :2] = self._perspective_transform(
                        bbox[j][:, :2], mat_output
                    )
                    bbox[j][:, 2:] = self._perspective_transform(
                        bbox[j][:, 2:], mat_output
                    )
                    bbox[j][:, [0, 2]] = np.clip(bbox[j][:, [0, 2]], 0, _output_size - 1)
                    bbox[j][:, [1, 3]] = np.clip(bbox[j][:, [1, 3]], 0, _output_size - 1)

            mat_input = self._get_affine_matrix(
                center, scale, (self.input_size, self.input_size), aug_rot, self.perspective
            )

            images[i] = cv2.warpPerspective(image, mat_input, (self.input_size, self.input_size))

        return images, image_type, bbox
