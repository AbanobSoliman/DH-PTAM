import cv2
import feature_superpoint
import feature_r2d2

class Params(object):
    def __init__(self):
        self.pnp_min_measurements = 5
        self.pnp_max_iterations = 10
        self.init_min_points = 5

        self.local_window_size = 5
        self.ba_max_iterations = 30

        self.min_tracked_points_ratio = 0.5

        self.lc_min_inbetween_frames = 5  # frames
        self.lc_max_inbetween_distance = 3  # meters
        self.lc_embedding_distance = 22.0
        self.lc_inliers_threshold = 15
        self.lc_inliers_ratio = 0.5
        self.lc_distance_threshold = 2  # meters
        self.lc_max_iterations = 30

        self.ground = False

        self.view_camera_size = 1


class ParamsEuroc(Params):

    def __init__(self, config='GFTT-BRIEF'):
        super().__init__()

        if config == 'GFTT-BRIEF':
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=1000, minDistance=15.0,
                qualityLevel=0.001, useHarrisDetector=False)

            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)

        elif config == 'ORB-BRIEF':
            self.feature_detector = cv2.ORB_create(
                nfeatures=10000, scaleFactor=1.2, nlevels=7, edgeThreshold=7)

            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)

        else:
            raise NotImplementedError

        self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.matching_cell_size = 15  # pixels
        self.matching_neighborhood = 2
        self.matching_distance = 25

        self.frustum_near = 0.1  # meters
        self.frustum_far = 50.0

        self.lc_max_inbetween_distance = 4  # meters
        self.lc_distance_threshold = 1.5
        self.lc_embedding_distance = 22.0

        self.view_image_width = 400
        self.view_image_height = 250
        self.view_camera_width = 0.1
        self.view_viewpoint_x = 0
        self.view_viewpoint_y = -1
        self.view_viewpoint_z = -10
        self.view_viewpoint_f = 2000


class ParamsTUMVIE(Params):

    def __init__(self, scale, config='SP'):
        super().__init__()

        if config == 'GFTT-BRIEF':
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=10000, minDistance=1.0,
                qualityLevel=0.07, useHarrisDetector=False)

            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=True)  # cv2.xfeatures2d_BoostDesc.create()

        elif config == 'ORB-BRIEF':
            self.feature_detector = cv2.ORB_create(
                nfeatures=1000000, scaleFactor=1.2, nlevels=7, edgeThreshold=1)

            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=16, use_orientation=True)

        elif config == 'GFTT-ORB':
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=10000, minDistance=1.0,
                qualityLevel=0.01, useHarrisDetector=False)  # detect

            self.descriptor_extractor = cv2.ORB_create(
                nfeatures=2000, scaleFactor=1.2, nlevels=20, edgeThreshold=1, fastThreshold=1, patchSize=500)  # compute

        elif config == 'ORB-Boost':
            self.feature_detector = cv2.ORB_create(
                nfeatures=10000, scaleFactor=1.2, nlevels=8, edgeThreshold=1, fastThreshold=1, patchSize=500)  # detect

            self.descriptor_extractor = cv2.xfeatures2d_BoostDesc.create()  # compute

        elif config == 'SP':
            self.feature_detector = feature_superpoint.SuperPointFeature2D()  # detect
            self.descriptor_extractor = self.feature_detector  # compute

        elif config == 'R2D2':
            self.feature_detector = feature_r2d2.R2d2Feature2D(num_features=10000, scale_f=2**0.25)  # detect
            self.descriptor_extractor = self.feature_detector  # compute

        else:
            raise NotImplementedError

        if scale:
            # Large-scale motion.
            self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            self.matching_cell_size = 15  # pixels   15
            self.matching_neighborhood = 1.8   # 2
            self.matching_distance = 15   # 25
            self.frustum_near = 0.4  # meters   # 0.1
            self.frustum_far = 5.0    # to 100   #  30
        else:
            # Small-scale motion.
            self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            self.matching_cell_size = 15  # pixels
            self.matching_neighborhood = 1.8
            self.matching_distance = 15
            self.frustum_near = 0.9  # meters
            self.frustum_far = 10.0  # to 1000

        self.ground = True

        self.lc_max_inbetween_distance = 25
        self.lc_distance_threshold = 10
        self.lc_embedding_distance = 20.0

        self.view_image_width = 300
        self.view_image_height = 300
        self.view_camera_width = 0.1
        self.view_viewpoint_x = 0
        self.view_viewpoint_y = -1
        self.view_viewpoint_z = -10
        self.view_viewpoint_f = 2000


class ParamsVECtor(Params):

    def __init__(self, scale, config='SP'):
        super().__init__()

        if config == 'GFTT-BRIEF':
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=10000, minDistance=1.0,
                qualityLevel=0.07, useHarrisDetector=False)

            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=True)  # cv2.xfeatures2d_BoostDesc.create()

        elif config == 'ORB-BRIEF':
            self.feature_detector = cv2.ORB_create(
                nfeatures=1000000, scaleFactor=1.2, nlevels=7, edgeThreshold=1)

            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=16, use_orientation=True)

        elif config == 'GFTT-ORB':
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=10000, minDistance=1.0,
                qualityLevel=0.01, useHarrisDetector=False)  # detect

            self.descriptor_extractor = cv2.ORB_create(
                nfeatures=2000, scaleFactor=1.2, nlevels=20, edgeThreshold=1, fastThreshold=1, patchSize=500)  # compute

        elif config == 'ORB-Boost':
            self.feature_detector = cv2.ORB_create(
                nfeatures=10000, scaleFactor=1.2, nlevels=8, edgeThreshold=1, fastThreshold=1, patchSize=500)  # detect

            self.descriptor_extractor = cv2.xfeatures2d_BoostDesc.create()  # compute


        elif config == 'SP':

            self.feature_detector = feature_superpoint.SuperPointFeature2D()  # detect

            self.descriptor_extractor = self.feature_detector  # compute


        elif config == 'R2D2':

            self.feature_detector = feature_r2d2.R2d2Feature2D(num_features=4000, scale_f=2 ** 0.25)  # detect

            self.descriptor_extractor = self.feature_detector  # compute

        else:
            raise NotImplementedError

        if scale:
            # Large-scale motion.
            self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            self.matching_cell_size = 15  # pixels   15
            self.matching_neighborhood = 1.8   # 2
            self.matching_distance = 15   # 25
            self.frustum_near = 0.1  # meters   # 0.1
            self.frustum_far = 30.0    # to 100   #  30
        else:
            # Small-scale motion.
            self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            self.matching_cell_size = 15  # pixels
            self.matching_neighborhood = 1.8
            self.matching_distance = 15
            self.frustum_near = 0.1  # meters
            self.frustum_far = 5.0  # to 1000

        self.ground = True

        self.lc_max_inbetween_distance = 25
        self.lc_distance_threshold = 10
        self.lc_embedding_distance = 20.0

        self.view_image_width = 300
        self.view_image_height = 300
        self.view_camera_width = 0.1
        self.view_viewpoint_x = 0
        self.view_viewpoint_y = -1
        self.view_viewpoint_z = -10
        self.view_viewpoint_f = 2000


class ParamsKITTI(Params):
    def __init__(self, config='GFTT-BRIEF'):
        super().__init__()

        if config == 'GFTT-BRIEF':
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=1000, minDistance=12.0,
                qualityLevel=0.001, useHarrisDetector=False)

            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)

        elif config == 'GFTT-BRISK':
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=2000, minDistance=15.0,
                qualityLevel=0.01, useHarrisDetector=False)

            self.descriptor_extractor = cv2.BRISK_create()

        elif config == 'ORB-ORB':
            self.feature_detector = cv2.ORB_create(
                nfeatures=1000, scaleFactor=1.2, nlevels=1, edgeThreshold=31)
            self.descriptor_extractor = self.feature_detector

        else:
            raise NotImplementedError

        self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.matching_cell_size = 15  # pixels
        self.matching_neighborhood = 3
        self.matching_distance = 30

        self.frustum_near = 0.1  # meters
        self.frustum_far = 1000.0

        self.ground = True

        self.lc_max_inbetween_distance = 50
        self.lc_distance_threshold = 15
        self.lc_embedding_distance = 20.0

        self.view_image_width = 400
        self.view_image_height = 130
        self.view_camera_width = 0.75
        self.view_viewpoint_x = 0
        self.view_viewpoint_y = -500  # -10
        self.view_viewpoint_z = -100  # -0.1
        self.view_viewpoint_f = 2000
