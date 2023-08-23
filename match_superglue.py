from pathlib import Path
import argparse, sys, cv2
import random
import numpy as np
import matplotlib.cm as cm
import torch

sys.path.insert(0, './features/SuperGluePretrainedNetwork/models')
from matching import Matching

torch.set_grad_enabled(False)


class SuperGlueMatcher(object):

    def __init__(self):
        self.matches = SuperMatcher()
        # Load the SuperPoint and SuperGlue models.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Running inference on device \"{}\"'.format(self.device))
        self.config = {
            'superpoint': {
                'descriptor_dim': 256,
                'nms_radius': 7,
                'keypoint_threshold': 0.07,
                'max_keypoints': -1,
                'remove_borders': 4,
            },
            'superglue': {
                'descriptor_dim': 256,
                'weights': 'indoor',
                'keypoint_encoder': [32, 64, 128, 256],
                'GNN_layers': ['self', 'cross'] * 9,
                'sinkhorn_iterations': 100,
                'match_threshold': 0.3,
            }
        }
        self.matching = Matching(self.config).eval().to(self.device)
        self.img0 = None
        self.img1 = None

    def frame2tensor(self, frame, device):
        return torch.from_numpy(frame / 255.).float()[None, None].to(device)

    def read_image(self, img, device, rotation):
        image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if image is None:
            return None, None, None
        w, h = image.shape[1], image.shape[0]
        scales = (float(w) / float(w), float(h) / float(h))

        if rotation != 0:
            image = np.rot90(image, k=rotation)
            if rotation % 2:
                scales = scales[::-1]

        inp = self.frame2tensor(image, device)
        return image, inp, scales

    def draw_keypoints(self, kps1_sizes=None, kps2_sizes=None):
        img3 = cv2.hconcat([self.img0, self.img1])
        h1, w1 = self.img0.shape[:2]
        N = len(self.matches.left["keypoints"])
        default_size = 2
        if kps1_sizes is None:
            kps1_sizes = np.ones(N, dtype=np.int32) * default_size
        if kps2_sizes is None:
            kps2_sizes = np.ones(N, dtype=np.int32) * default_size
        for i, pts in enumerate(zip(cv2.KeyPoint.convert(self.matches.left["keypoints"]),
                                    cv2.KeyPoint.convert(self.matches.right["keypoints"]))):
            p1, p2 = np.rint(pts).astype(int)
            a, b = p1.ravel()
            c, d = p2.ravel()
            size1 = kps1_sizes[i]
            size2 = kps2_sizes[i]
            color = tuple(np.random.randint(0, 255, 3).tolist())
            # cv2.line(img3, (a,b),(c,d), color, 1)    # optic flow style
            cv2.line(img3, (a, b), (c + w1, d), color, 1)  # join corresponding points
            cv2.circle(img3, (a, b), 2, color, -1)
            cv2.circle(img3, (a, b), color=(0, 255, 0), radius=size1, thickness=1)  # draw keypoint size as a circle
            cv2.circle(img3, (c + w1, d), 2, color, -1)
            cv2.circle(img3, (c + w1, d), color=(0, 255, 0), radius=size2,
                       thickness=1)  # draw keypoint size as a circle
        return img3

    def SuperMatch(self, img0, img1):
        # Load the image pair.
        self.img0, self.img1 = img0, img1
        image0, inp0, scales0 = self.read_image(self.img0, self.device, 0)
        image1, inp1, scales1 = self.read_image(self.img1, self.device, 0)
        # Perform the matching.
        pred = self.matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        conf0, conf1 = pred['scores0'], pred['scores1']
        desc0, desc1 = pred['descriptors0'].T, pred['descriptors1'].T
        matches0, conf = pred['matches0'], pred['matching_scores0']
        # Keep the matching keypoints.
        valid = matches0 > -1
        self.matches.left = {'descriptors': desc0[valid], 'keypoints': cv2.KeyPoint_convert(kpts0[valid])}
        self.matches.right = {'descriptors': desc1[matches0[valid]],
                              'keypoints': cv2.KeyPoint_convert(kpts1[matches0[valid]])}


class SuperMatcher(object):
    def __int__(self):
        self.left = None
        self.right = None
