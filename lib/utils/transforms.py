from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2


def transform_preds(coords, center, scale, output_size):

    target_coords = coords.copy()
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
        target_coords[p, 2:4] = affine_transform(coords[p, 2:4], trans)

        target_coords[p, [0, 2]] = np.clip(target_coords[p, [0, 2]], 0, output_size[0] - 1)
        target_coords[p, [1, 3]] = np.clip(target_coords[p, [1, 3]], 0, output_size[1] - 1)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def resize(image, input_size):
    h, w, _ = image.shape

    center = np.array([int(w/2.0+0.5), int(h/2.0+0.5)])
    if w < h:
        w_resized = input_size
        h_resized = int((input_size / w * h + 63) // 64 * 64)
        scale_w = w / 200.0
        scale_h = h_resized / w_resized * w / 200.0
    else:
        h_resized = input_size
        w_resized = int((input_size / h * w + 63) // 64 * 64)
        scale_h = h / 200.0
        scale_w = w_resized / h_resized * h / 200.0

    scale = np.array([scale_w, scale_h])
    trans = get_affine_transform(center, scale, 0, (w_resized, h_resized))

    image_resized = cv2.warpAffine(
        image,
        trans,
        (int(w_resized), int(h_resized))
    )

    return image_resized, center, scale


def get_multi_scale_size(image, input_size, current_scale, min_scale, saved_size):
    h, w, _ = image.shape
    center = np.array([int(w / 2.0 + 0.5), int(h / 2.0 + 0.5)])

    # calculate the size for min_scale
    min_input_size = int((min_scale * input_size + 63)//64 * 64)
    if saved_size:
        cond = saved_size[0] < saved_size[1]
    else:
        cond = w < h

    if cond:
        w_resized = int(min_input_size * current_scale / min_scale)
        h_resized = int(
            int((min_input_size/w*h+63)//64*64)*current_scale/min_scale
        )
        scale_w = w / 200.0
        scale_h = h_resized / w_resized * w / 200.0
    else:
        h_resized = int(min_input_size * current_scale / min_scale)
        w_resized = int(
            int((min_input_size/h*w+63)//64*64)*current_scale/min_scale
        )
        scale_h = h / 200.0
        scale_w = w_resized / h_resized * h / 200.0

    return (w_resized, h_resized), center, np.array([scale_w, scale_h])


def resize_align_multi_scale(image, input_size, current_scale, min_scale, saved_size):

    size_resized, center, scale = get_multi_scale_size(
        image, input_size, current_scale, min_scale, saved_size
    )

    trans = get_affine_transform(center, scale, 0, size_resized)
    if saved_size:
        trans[0, 2] += 0.5 * (saved_size[0] - size_resized[0])
    image_resized = cv2.warpAffine(
        image,
        trans,
        saved_size if saved_size else size_resized
    )

    return image_resized, center, scale, trans


def get_final_preds(ans, center, scale, heatmap_size):
    final_results = []

    for person in ans[0]:
        bbox = transform_preds(person, center, scale, heatmap_size)
        final_results.append(bbox)

    return final_results
