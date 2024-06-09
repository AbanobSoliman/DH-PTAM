import g2o
import cv2
import os
import sys
import glob
import math
import argparse
from datetime import date
from datetime import datetime
import numpy as np
import scipy
from scipy.spatial.transform import Rotation
from threading import Thread
from components import Camera
from components import StereoFrame
from feature import ImageFeature
from params import *
from dataset import KITTIOdometry, EuRoCDataset, TUMVIEDataset
import matplotlib.pyplot as mpplot
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from itertools import chain
from collections import defaultdict
from covisibility import CovisibilityGraph
from optimization import BundleAdjustment
from mapping import Mapping
from mapping import MappingThread
from components import Measurement
from motion import MotionModel
from loopclosing import LoopClosing
from natsort import natsorted, ns
from progress.bar import Bar
from tqdm import tqdm
import hdf5plugin
import h5py
from pathlib import Path
from skimage import data
from skimage.util import img_as_ubyte
from skimage.morphology import erosion, dilation, opening, closing, white_tophat, diameter_closing
from skimage.morphology import black_tophat, skeletonize, convex_hull_image, diameter_opening
from skimage.morphology import disk, diamond
from dataset_config import *

sys.path.insert(0, './thirdparty/DSEC/scripts/')
sys.path.insert(0, './thirdparty/DSEC/ip_basic/ip_basic')
from visualization.eventreader import EventReader
from utils.eventslicer import EventSlicer
from numba import jit

import warnings
warnings.filterwarnings("ignore")

# import match_superglue

# Super resolution
sr = cv2.dnn_superres.DnnSuperResImpl_create()
path = "./superres/ESPCN_x4.pb"
sr.readModel(path)
sr.setModel("espcn", 4)  # set the model by passing the value and the upsampling ratio


def slice_events_dict(events_dict, t_start, t_end):
    """
    Slice the events dictionary based on the given start and end time.

    Parameters:
    - events_dict (dict): The original events dictionary with keys ['x', 'y', 't', 'p'].
    - t_start (float): Start time for the slice.
    - t_end (float): End time for the slice.

    Returns:
    - dict: A dictionary with sliced events.
    """
    # Find the indices for the given time range
    t_array = events_dict['t']
    start_idx = next((i for i, t in enumerate(t_array) if t >= t_start), None)
    end_idx = next((i for i, t in enumerate(t_array) if t > t_end), len(t_array)) - 1

    if start_idx is None:
        return {key: [] for key in events_dict}

    # Slice the dictionary
    sliced_dict = {
        'x': events_dict['x'][start_idx:end_idx + 1],
        'y': events_dict['y'][start_idx:end_idx + 1],
        't': events_dict['t'][start_idx:end_idx + 1],
        'p': events_dict['p'][start_idx:end_idx + 1]
    }

    return sliced_dict


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def quat_norm(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if mag2 > tolerance:
        mag = math.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return np.array(v).astype(float)


def align_frames(img1, img2):
    # 1 to be aligned
    # 2 ref image
    height, width = img2.shape
    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(50000)
    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)
    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)
    # Sort matches on the basis of their Hamming distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 0.9)]
    no_of_matches = len(matches)
    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt
    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1, homography, (width, height))
    return transformed_img


def load_images_from_folder(folder):
    img_list = natsorted(folder, key=lambda y: y.lower())
    return img_list


def get_black_white_indices(hist, tot_count, black_count, white_count):
    '''Blacking and Whiting out indices same as color balance'''

    black_ind = 0
    white_ind = 255
    co = 0
    for i in range(len(hist)):
        co += hist[i]
        if co > black_count:
            black_ind = i
            break

    co = 0
    for i in range(len(hist) - 1, -1, -1):
        co += hist[i]
        if co > (tot_count - white_count):
            white_ind = i
            break

    return [black_ind, white_ind]


def contrast_stretch(img, black_point, white_point):
    '''Contrast stretch image with black and white cap'''

    tot_count = img.shape[0] * img.shape[1]
    black_count = tot_count * black_point / 100
    white_count = tot_count * white_point / 100
    ch_hists = []
    # calculate histogram for each channel
    for ch in cv2.split(img):
        ch_hists.append(cv2.calcHist([ch], [0], None, [256], (0, 256)).flatten().tolist())

    # get black and white percentage indices
    black_white_indices = []
    for hist in ch_hists:
        black_white_indices.append(get_black_white_indices(hist, tot_count, black_count, white_count))

    stretch_map = np.zeros((3, 256), dtype='uint8')

    # stretch histogram
    for curr_ch in range(len(black_white_indices)):
        black_ind, white_ind = black_white_indices[curr_ch]
        for i in range(stretch_map.shape[1]):
            if i < black_ind:
                stretch_map[curr_ch][i] = 0
            else:
                if i > white_ind:
                    stretch_map[curr_ch][i] = 255
                else:
                    if (white_ind - black_ind) > 0:
                        stretch_map[curr_ch][i] = round((i - black_ind) / (white_ind - black_ind)) * 255
                    else:
                        stretch_map[curr_ch][i] = 0

    # stretch image
    ch_stretch = []
    for i, ch in enumerate(cv2.split(img)):
        ch_stretch.append(cv2.LUT(ch, stretch_map[i]))

    return cv2.merge(ch_stretch)


def fast_gaussian_blur(img, ksize, sigma):
    '''Gussian blur using linear separable property of Gaussian distribution'''

    kernel_1d = cv2.getGaussianKernel(ksize, sigma)
    return cv2.sepFilter2D(img, -1, kernel_1d, kernel_1d)


def gamma(img, gamma_value):
    '''Gamma correction of image'''

    i_gamma = 1 / gamma_value
    lut = np.array([((i / 255) ** i_gamma) * 255 for i in np.arange(0, 256)], dtype='uint8')
    return cv2.LUT(img, lut)


def color_balance(img, low_per, high_per):
    '''Contrast stretch image by histogram equilization with black and white cap'''

    tot_pix = img.shape[1] * img.shape[0]
    # no.of pixels to black-out and white-out
    low_count = tot_pix * low_per / 100
    high_count = tot_pix * (100 - high_per) / 100

    cs_img = []
    # for each channel, apply contrast-stretch
    for ch in cv2.split(img):
        # cummulative histogram sum of channel
        cum_hist_sum = np.cumsum(cv2.calcHist([ch], [0], None, [256], (0, 256)))

        # find indices for blacking and whiting out pixels
        li, hi = np.searchsorted(cum_hist_sum, (low_count, high_count))
        if (li == hi):
            cs_img.append(ch)
            continue
        # lut with min-max normalization for [0-255] bins
        lut = np.array([0 if i < li
                        else (255 if i > hi else round((i - li) / (hi - li) * 255))
                        for i in np.arange(0, 256)], dtype='uint8')
        # constrast-stretch channel
        cs_ch = cv2.LUT(ch, lut)
        cs_img.append(cs_ch)

    return cv2.merge(cs_img)


def enhance_image(imgLowRes):
    img = cv2.resize(imgLowRes, (2 * imgLowRes.shape[1], 2 * imgLowRes.shape[0]), cv2.INTER_CUBIC)
    contrast_stretch_img = contrast_stretch(img, 1, 70.5)
    blured = cv2.medianBlur(contrast_stretch_img, 3)  # img = cv2.bilateralFilter(img, 3, 3, 3)  # 7 supres
    gamma_img = gamma(blured, 1.1)
    img = color_balance(gamma_img, 2, 1)
    img_down = cv2.resize(img, (imgLowRes.shape[1], imgLowRes.shape[0]), cv2.INTER_AREA)
    img_down = cv2.fastNlMeansDenoisingColored(img_down, None, 5, 5, 7, 2)
    img_thresh = closing(cv2.cvtColor(img_down, cv2.COLOR_RGB2GRAY), disk(7))
    img_thresh = cv2.adaptiveThreshold(img_thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
    img_down[img_thresh == [0]] = [0, 0, 0]
    return cv2.cvtColor(img_down, cv2.COLOR_RGB2GRAY)


def E3CT_preprocessing(xypt, output_array, total_tbins_delta_t, downsampling_factor=0, reset=True):
    """
    Args:
        xypt (events): structured array containing events
        output_array: ndarray
        total_tbins_delta_t (int): total duration of the time slice
        downsampling_factor (int): parameter used to reduce the spatial dimension of the obtained feature.
                                   in practice, the original coordinates will be multiplied by 2**(-downsampling_factor).
        reset (boolean): whether to reset the output_array to 0 or not.
    """
    num_tbins = output_array.shape[0]
    dt = int(total_tbins_delta_t / num_tbins)

    if reset:
        output_array[...] = 0

    ti = 0  # time bins starts at 0
    bin_threshold_int = int(math.ceil(dt))  # integer bound for the time bin
    for i in range(xypt.shape[0]):
        x, y, p, t = xypt['x'][i] >> downsampling_factor, xypt['y'][i] >> downsampling_factor, xypt['p'][i], \
            xypt['t'][i]  # get the event information, scaled if needed
        # we compute the time bin
        if t >= bin_threshold_int and ti + 1 < num_tbins:
            ti = int(t // dt)
            bin_threshold_int = int(math.ceil((ti + 1) * dt))
        output_array[ti, p, y, x] = 1  # set one for each event we receive


# the jit function (Just-in-Time) can be used as a decorator or directly like this
numba_E3CT_preprocessing = jit(E3CT_preprocessing, nopython=True)


def e3ct_postprocess(imgLowRes):
    # Filter the channels (median / bilateral)
    # img = sr.upsample(imgLowRes)  # upscale the input image and threshold it
    img = cv2.resize(imgLowRes, (2 * imgLowRes.shape[1], 2 * imgLowRes.shape[0]), cv2.INTER_CUBIC)
    img_down = cv2.resize(img, (imgLowRes.shape[1], imgLowRes.shape[0]), cv2.INTER_AREA)
    img_down = cv2.fastNlMeansDenoising(img_down, None, 5, 5, 5)
    img_thresh = cv2.adaptiveThreshold(img_down, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
    img_down[img_thresh == [255]] = 0
    # Stereo Undistort and Rectify images
    return img_down


# the jit function (Just-in-Time) can be used as a decorator or directly like this
numba_e3ct_postprocess = jit(e3ct_postprocess)


class e3ct_create(object):
    def __init__(self, delta_t, events_pre, height, width, delta, xmap1, ymap1, aps0, Kc, Tce, Ke_inv, delta_uv):
        self.delta_t = delta_t
        self.events_pre = events_pre
        self.height = height
        self.width = width
        self.delta = delta
        self.xmap = xmap1
        self.ymap = ymap1
        self.aps = aps0
        self.h_c, self.w_c = self.aps.shape
        self.fusion = None
        self.E3CT = None
        self.E3CT_rect = None
        self.Kc = Kc
        self.Ke_inv = Ke_inv
        self.Tce = Tce
        self.delta_uv = delta_uv

    def construct(self):
        events = np.zeros(len(self.events_pre[:, 2]), dtype=[('x', 'u2'), ('y', 'u2'), ('p', 'i2'),
                                                             ('t', 'i8')])  # create empty structured np array
        events['t'] = self.events_pre[:, 2]
        events['p'] = self.events_pre[:, 3]
        events['x'] = self.events_pre[:, 0]
        events['y'] = self.events_pre[:, 1]
        events['t'] -= int(events['t'][0])
        # Exponential Decay Time Surface (alpha-kernel) then Event Spike Tensor (trilinear-kernel)
        volume = np.zeros((6, 2, self.height, self.width), dtype=np.float64)
        numba_E3CT_preprocessing(events, volume, self.delta_t * 1e3,
                                 reset=True)  # delta_t from milli seconds to micro seconds
        sum = (volume.sum(axis=0)).sum(axis=0)
        img = (sum * 255).astype(np.uint8)
        img_down = numba_e3ct_postprocess(img)
        self.E3CT = cv2.remap(img_down, self.xmap, self.ymap, cv2.INTER_CUBIC, borderValue=(0, 0, 0, 0))

    def fuse(self, alpha):
        E = np.ones_like(self.aps)
        self.construct()
        xe, ye = np.where(self.E3CT.astype(np.uint8) == 255)
        fe = np.linalg.norm([np.linalg.inv(self.Ke_inv)[0, 0], np.linalg.inv(self.Ke_inv)[1, 1]])
        pix_e = np.array([xe, ye, fe * np.ones_like(xe)])
        pix_c = self.Kc @ (
                np.hstack([self.Tce[:3, :3], -self.Tce[:3, :3] @ self.Tce[:3, -1].reshape(3, 1)]) @ np.vstack(
            [self.Ke_inv @ pix_e, np.ones_like(xe)]))
        if pix_c.shape[1] == 0:
            print("No E3CT to fuse with this frame!")
            self.E3CT_rect = (255 * E).astype(np.uint8)  # cv2.medianBlur((255 * E).astype(np.uint8), 7)
            self.fusion = cv2.addWeighted(self.aps, 1 - alpha, self.E3CT_rect, alpha, 0.0)
            return
        d_fec = abs(fe - np.mean(pix_c[2, :]))
        pix_c += np.array([d_fec, d_fec, d_fec]).reshape(3, 1)
        pix_c -= np.array([np.min(pix_c[0, :]), np.min(pix_c[1, :]), np.min(pix_c[2, :])]).reshape(3, 1)
        xc = pix_c[0].astype(int) + self.delta_uv[0]
        yc = pix_c[1].astype(int) + self.delta_uv[1]
        mask = np.logical_and(np.logical_and(pix_c[0, :] >= 0, pix_c[0, :] <= self.aps.shape[0]),
                              np.logical_and(pix_c[1, :] >= 0, pix_c[1, :] <= self.aps.shape[1]))
        xc = xc[mask].astype(int)
        yc = yc[mask].astype(int)
        E[xc, yc] = 0
        self.E3CT_rect = (255 * E).astype(np.uint8)
        self.aps = self.aps * (1 - alpha)
        self.aps[xc, yc] = 255
        self.fusion = (self.aps).astype(np.uint8)


class Tracking(object):
    def __init__(self, params):
        self.optimizer = BundleAdjustment()
        self.min_measurements = params.pnp_min_measurements
        self.max_iterations = params.pnp_max_iterations

    def refine_pose(self, pose, cam, measurements):
        assert len(measurements) >= self.min_measurements, (
            'Not enough points')

        self.optimizer.clear()
        self.optimizer.add_pose(0, pose, cam, fixed=False)

        for i, m in enumerate(measurements):
            self.optimizer.add_point(i, m.mappoint.position, fixed=True)
            self.optimizer.add_edge(0, i, 0, m)

        self.optimizer.optimize(self.max_iterations)
        return self.optimizer.get_pose(0)


class SPTAM(object):
    def __init__(self, params):
        self.params = params

        self.tracker = Tracking(params)
        self.motion_model = MotionModel()

        self.graph = CovisibilityGraph()
        self.mapping = MappingThread(self.graph, params)

        self.loop_closing = LoopClosing(self, params)
        self.loop_correction = None

        self.reference = None  # reference keyframe
        self.preceding = None  # last keyframe
        self.current = None  # current frame
        self.status = defaultdict(bool)

    def stop(self):
        self.mapping.stop()
        if self.loop_closing is not None:
            self.loop_closing.stop()

    def initialize(self, frame):
        mappoints, measurements = frame.triangulate()
        assert len(mappoints) >= self.params.init_min_points, (
            'Not enough points to initialize map.')

        keyframe = frame.to_keyframe()
        keyframe.set_fixed(True)
        self.graph.add_keyframe(keyframe)
        self.mapping.add_measurements(keyframe, mappoints, measurements)
        if self.loop_closing is not None:
            self.loop_closing.add_keyframe(keyframe)

        self.reference = keyframe
        self.preceding = keyframe
        self.current = keyframe
        self.status['initialized'] = True

        self.motion_model.update_pose(
            frame.timestamp, frame.position, frame.orientation)

    def track(self, frame):
        while self.is_paused():
            time.sleep(1e-4)
        self.set_tracking(True)

        self.current = frame
        print('Tracking:', frame.idx, ' <- ', self.reference.id, self.reference.idx)

        predicted_pose, _ = self.motion_model.predict_pose(frame.timestamp)
        frame.update_pose(predicted_pose)

        if self.loop_closing is not None:
            if self.loop_correction is not None:
                estimated_pose = g2o.Isometry3d(
                    frame.orientation,
                    frame.position)
                estimated_pose = estimated_pose * self.loop_correction
                frame.update_pose(estimated_pose)
                self.motion_model.apply_correction(self.loop_correction)
                self.loop_correction = None

        local_mappoints = self.filter_points(frame)
        measurements = frame.match_mappoints(
            local_mappoints, Measurement.Source.TRACKING)

        print('measurements:', len(measurements), '   ', len(local_mappoints))

        tracked_map = set()
        for m in measurements:
            mappoint = m.mappoint
            mappoint.update_descriptor(m.get_descriptor())
            mappoint.increase_measurement_count()
            tracked_map.add(mappoint)

        try:
            self.reference = self.graph.get_reference_frame(tracked_map)

            pose = self.tracker.refine_pose(frame.pose, frame.cam, measurements)
            frame.update_pose(pose)
            self.motion_model.update_pose(
                frame.timestamp, pose.position(), pose.orientation())
            tracking_is_ok = True
        except:
            tracking_is_ok = False
            print('tracking failed!!!')

        if tracking_is_ok and self.should_be_keyframe(frame, measurements):
            print('new keyframe', frame.idx)
            keyframe = frame.to_keyframe()
            keyframe.update_reference(self.reference)
            keyframe.update_preceding(self.preceding)

            self.mapping.add_keyframe(keyframe, measurements)
            if self.loop_closing is not None:
                self.loop_closing.add_keyframe(keyframe)
            self.preceding = keyframe

        self.set_tracking(False)

    def filter_points(self, frame):
        local_mappoints = self.graph.get_local_map_v2(
            [self.preceding, self.reference])[0]

        can_view = frame.can_view(local_mappoints)
        print('filter points:', len(local_mappoints), can_view.sum(),
              len(self.preceding.mappoints()),
              len(self.reference.mappoints()))

        checked = set()
        filtered = []
        for i in np.where(can_view)[0]:
            pt = local_mappoints[i]
            if pt.is_bad():
                continue
            pt.increase_projection_count()
            filtered.append(pt)
            checked.add(pt)

        for reference in set([self.preceding, self.reference]):
            for pt in reference.mappoints():  # neglect can_view test
                if pt in checked or pt.is_bad():
                    continue
                pt.increase_projection_count()
                filtered.append(pt)

        return filtered

    def should_be_keyframe(self, frame, measurements):
        if self.adding_keyframes_stopped():
            return False

        n_matches = len(measurements)
        n_matches_ref = len(self.reference.measurements())

        print('keyframe check:', n_matches, '   ', n_matches_ref)

        return ((n_matches / n_matches_ref) <
                self.params.min_tracked_points_ratio) or n_matches < 20

    def set_loop_correction(self, T):
        self.loop_correction = T

    def is_initialized(self):
        print("Map Reconstructed Successfully!")
        return self.status['initialized']

    def pause(self):
        self.status['paused'] = True

    def unpause(self):
        self.status['paused'] = False

    def is_paused(self):
        return self.status['paused']

    def is_tracking(self):
        return self.status['tracking']

    def set_tracking(self, status):
        self.status['tracking'] = status

    def stop_adding_keyframes(self):
        self.status['adding_keyframes_stopped'] = True

    def resume_adding_keyframes(self):
        self.status['adding_keyframes_stopped'] = False

    def adding_keyframes_stopped(self):
        return self.status['adding_keyframes_stopped']
