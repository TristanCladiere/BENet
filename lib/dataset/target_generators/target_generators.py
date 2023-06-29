from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class BBGenerator:
    def __init__(self, output_res, max_num_people, sigma=-1):
        self.output_res = output_res
        self.max_num_people = max_num_people
        if sigma < 0:
            sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, bbox):
        hms = np.zeros((self.output_res, self.output_res),
                       dtype=np.float32)
        hw = np.zeros((self.max_num_people, 2, 3))
        sigma = self.sigma
        for i, p in enumerate(bbox):
            x_c, y_c = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
            dims = p[3] - p[1], p[2] - p[0]  # h, w
            for j in range(2):
                hw[i, j] = (dims[j], j * self.output_res**2 + int(y_c) * self.output_res + int(x_c), 1)

            ul = int(np.round(int(x_c) - 3 * sigma - 1)), int(np.round(int(y_c) - 3 * sigma - 1))
            br = int(np.round(int(x_c) + 3 * sigma + 2)), int(np.round(int(y_c) + 3 * sigma + 2))

            c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], self.output_res)
            aa, bb = max(0, ul[1]), min(br[1], self.output_res)
            hms[aa:bb, cc:dd] = np.maximum(
                hms[aa:bb, cc:dd], self.g[a:b, c:d])

        return hms.astype(np.float32), hw.astype(np.float32)


class EmotionsGenerator:
    def __init__(self, max_num_people, num_emo, output_res):
        self.max_num_people = max_num_people
        self.output_res = output_res
        self.num_emo = num_emo

    def __call__(self, bbox, emotions):
        visible_emo = np.zeros((self.max_num_people, self.num_emo, 3))
        output_res = self.output_res
        if emotions.any():
            for i, p in enumerate(bbox):
                x1, y1, x2, y2 = p
                x_c = int((x1+x2)/2)
                y_c = int((y1+y2)/2)
                for emo_idx, emo in enumerate(emotions[i]):
                    visible_emo[i][emo_idx] = (emo, emo_idx * output_res**2 + y_c * output_res + x_c, 1)

        return visible_emo
