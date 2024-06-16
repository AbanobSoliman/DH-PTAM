import numpy as np

from backend_FE_exp import *
sys.path.insert(0, './thirdparty/DSEC/scripts/')
sys.path.insert(0, './thirdparty/DSEC/ip_basic/ip_basic')
from visualization.eventreader import EventReader
from utils.eventslicer import EventSlicer

def load_images_from_folder(folder):
    img_list = natsorted(folder, key=lambda y: y.lower())
    return img_list


def Dataset_loading():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="baseline")  # tum or vector or baseline
    parser.add_argument("--dataset_name",
                        default="desk")  # mocap-desk2   ,   mocap-6dof   ,   bike-easy   ,   floor2-dark , corner_slow1,  school_scooter1, corridors_walk1
    parser.add_argument("--scale", default=0, type=int)  # Motion: 1: Large    0: small
    parser.add_argument("--skip", default=2, type=int)  # Skip the first X frames (45 for TUM-VIE, 1 for VECtor)
    parser.add_argument("--visualize", default=True, type=bool)
    parser.add_argument("--beta_lim", default=0.3, type=float)  # The contribution of events on the standard camera frame
    parser.add_argument("--dataset_path", default="/media/abanob/My Passport/Dense_mapping/3Ms")
    parser.add_argument("--feature", default="SP")  # R2D2 or SP (SuperPoint)
    args = parser.parse_args()

    dataset_type = args.dataset_type
    dataset_name = args.dataset_name
    scale = args.scale
    dataset_path = args.dataset_path

    if dataset_type == 'tum':

        parser.add_argument("--cam0", help="left DAVIS sensor grayscale frames",
                            default="%s/TUM-VIE/%s/data/left_images" % (dataset_path,
                                dataset_name))
        parser.add_argument("--cam1", help="right DAVIS sensor grayscale frames",
                            default="%s/TUM-VIE/%s/data/right_images" % (dataset_path,
                                dataset_name))
        parser.add_argument("--cam0ts", help="left DAVIS sensor grayscale frames timestamps us",
                            default="%s/TUM-VIE/%s/data/left_images/image_timestamps_left.txt" % (dataset_path,
                                dataset_name))
        parser.add_argument("--cam1ts", help="right DAVIS sensor grayscale frames timestamps us",
                            default="%s/TUM-VIE/%s/data/right_images/image_timestamps_right.txt" % (dataset_path,
                                dataset_name))
        parser.add_argument("--cam0exp", help="left DAVIS sensor grayscale frames exposure us",
                            default="%s/TUM-VIE/%s/data/left_images/image_exposures_left.txt" % (dataset_path,
                                dataset_name))
        parser.add_argument("--cam1exp", help="right DAVIS sensor grayscale frames exposure us",
                            default="%s/TUM-VIE/%s/data/right_images/image_exposures_right.txt" % (dataset_path,
                                dataset_name))
        parser.add_argument("--event_file1", help="Events in 'events' are of type (x, y, t, p)"
                                                  "ms_to_idx is same as in DSEC file format, which allows accessing event chunks using time index",
                            default="%s/TUM-VIE/%s/%s-events_left.h5" % (dataset_path,
                                dataset_name, dataset_name))
        parser.add_argument("--event_file2", help="Events in 'events' are of type (x, y, t, p)"
                                                  "ms_to_idx is same as in DSEC file format, which allows accessing event chunks using time index",
                            default="%s/TUM-VIE/%s/%s-events_right.h5" % (dataset_path,
                                dataset_name, dataset_name))
        parser.add_argument("--gt_poses", help="Events in 'events' are of type (x, y, t, p)"
                                               "ms_to_idx is same as in DSEC file format, which allows accessing event chunks using time index",
                            default="%s/TUM-VIE/%s/data/mocap_data.txt" % (dataset_path,
                                dataset_name))
        args = parser.parse_args()

        gt_poses_file = Path(args.gt_poses)
        gt_poses = np.loadtxt(gt_poses_file)
        cam0_times_file = Path(args.cam0ts)  # In micro seconds NOT nano !!
        cam0_times = np.loadtxt(cam0_times_file)
        cam1_times_file = Path(args.cam1ts)
        cam1_times = np.loadtxt(cam1_times_file)
        cam0_exp_file = Path(args.cam0exp)
        cam0_exp = np.loadtxt(cam0_exp_file)
        cam0_times += (0.5 * cam0_exp)
        cam1_exp_file = Path(args.cam1exp)
        cam1_exp = np.loadtxt(cam1_exp_file)
        cam1_times += (0.5 * cam1_exp)
        cam0_images = load_images_from_folder(glob.glob(os.path.join(Path(args.cam0), '*.jpg')))
        cam1_images = load_images_from_folder(glob.glob(os.path.join(Path(args.cam1), '*.jpg')))
        rawfileL = Path(args.event_file1)  # TUM-VIE  1, 2
        rawfileR = Path(args.event_file2)  # TUM-VIE  1, 2
        h, w = 720, 1280  # TUM-VIE DVS 720, 1280
        h_c, w_c = 1024, 1024  # TUM-VIE APS 1024, 1024

        freq_cam = 1 / np.mean(abs(cam0_times[:-1] - cam0_times[1:]) * 1e-6)
        dt_ms = 1000 / freq_cam  # in msec (Events Accumulation Time calculated to correspond to Global Shutter frames)
        delta = 30  # Time Surface Exponential Decay dT (30 msec is good)

        # DVS stereo rectification
        kcam_L = np.array(
            [[1049.5830934616608, 0, 634.7184038833433], [0, 1049.4229746040553, 263.46974530961836], [0, 0, 1]])
        pcam_l = np.array([-0.011519655713574485, -0.006222183183004903, 0.0021682612342850954, -0.0023528623774744806])
        kcam_R = np.array(
            [[1047.6678159506512, 0, 652.9082479181607], [0, 1047.523065288852, 260.5410079204179], [0, 0, 1]])
        pcam_R = np.array([-0.012461068584135461, 0.006075037311499091, 0.0035661921345332574, -0.008888043680457312])
        T_iL = np.eye(4)
        T_iL[:3, :3] = Rotation.from_quat(
            [-0.7015384510188221, 0.7125976574153885, 0.006516107167940321, 0.002433256169906342]).as_matrix()
        T_iL[:3, -1] = [-0.034803406191293906, 0.05971773350374604, -0.03694727254557562]
        T_iR = np.eye(4)
        T_iR[:3, :3] = Rotation.from_quat(
            [-0.7005187205007841, 0.7135042770180386, 0.009433399777504169, -0.009807133809032515]).as_matrix()
        T_iR[:3, -1] = [-0.03514747631374558, -0.05874695634395075, -0.036298272190308914]
        T_RL = np.linalg.inv(np.linalg.inv(T_iL) @ T_iR)
        R = T_RL[:3, :3]
        T = T_RL[:3, -1]
        Te = T
        R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(kcam_L, pcam_l, kcam_R, pcam_R, (w, h), R, T)
        xmap1, ymap1 = cv2.initUndistortRectifyMap(kcam_L, pcam_l, R1, kcam_L, (w, h), cv2.CV_32FC1)
        xmap2, ymap2 = cv2.initUndistortRectifyMap(kcam_R, pcam_R, R2, kcam_R, (w, h), cv2.CV_32FC1)
        Ke_L_inv = np.linalg.inv(kcam_L)
        Ke_L = kcam_L
        Ke_R_inv = np.linalg.inv(kcam_R)
        Tie_L = T_iL
        Tie_R = T_iR

        # APS stereo rectification
        kcam_L = np.array(
            [[747.3121949097032, 0, 490.8606682225008], [0, 747.1524375957724, 505.2373814853779], [0, 0, 1]])
        pcam_l = np.array([0.019702988705266412, 0.0019647288278805426, 0.0020858202586604944, -0.0009536922337319427])
        kcam_R = np.array(
            [[742.5207932360779, 0, 494.72989356499005], [0, 742.4219114252868, 513.9144417615198], [0, 0, 1]])
        pcam_R = np.array([0.019143394892292526, 0.0017469400418519364, 0.003535762629018563, -0.0014236433279599385])
        T_iL = np.eye(4)
        T_iL[:3, :3] = Rotation.from_quat(
            [-0.7073312004635668, 0.7068683247884341, -0.0015069364959480974, -0.00418012008915322]).as_matrix()
        T_iL[:3, -1] = [0.00961398490183905, 0.053927993466963454, -0.026555625688742424]
        T_iR = np.eye(4)
        T_iR[:3, :3] = Rotation.from_quat(
            [-0.7073932567310286, 0.7068137336302908, 0.0024961725648322422, -0.0017015891408379115]).as_matrix()
        T_iR[:3, -1] = [0.009377835983334585, -0.05549194484625431, -0.02642611704614006]
        T_RL = np.linalg.inv(np.linalg.inv(T_iL) @ T_iR)
        R = T_RL[:3, :3]
        T = T_RL[:3, -1]
        Tcam = T
        R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(kcam_L, pcam_l, kcam_R, pcam_R, (w_c, h_c), R, T)
        xmap3, ymap3 = cv2.initUndistortRectifyMap(kcam_L, pcam_l, R1, kcam_L, (w_c, h_c), cv2.CV_32FC1)
        xmap4, ymap4 = cv2.initUndistortRectifyMap(kcam_R, pcam_R, R2, kcam_R, (w_c, h_c), cv2.CV_32FC1)
        Kc_L = kcam_L
        Kc_R = kcam_R
        Tci_L = np.linalg.inv(T_iL)
        Tci_R = np.linalg.inv(T_iR)

        Tce_L = Tci_L @ Tie_L
        Tce_R = Tci_R @ Tie_R

        params = ParamsTUMVIE(scale, args.feature)

        h5file = h5py.File(rawfileL)
        print(h5file.keys())
        eventsl = h5file['events']
        print("Left Contains %d events" % (eventsl['t'].shape[0]))
        print("Left Event duration is %.7f seconds" % ((eventsl['t'][-1] - eventsl['t'][0]) * 1e-6))
        eventsL = EventSlicer(h5file)

        h5file = h5py.File(rawfileR)
        print(h5file.keys())
        eventsr = h5file['events']
        print("Right Contains %d events" % (eventsr['t'].shape[0]))
        print("Right Event duration is %.7f seconds" % ((eventsr['t'][-1] - eventsr['t'][0]) * 1e-6))
        eventsR = EventSlicer(h5file)

        delta_uv_L = (+355, +40)
        delta_uv_R = (+375, +45)

    if dataset_type == 'baseline':

        parser.add_argument("--cam0", help="left DAVIS sensor grayscale frames",
                            default="%s/Baseline/%s/data/left_images" % (dataset_path,
                                dataset_name))
        parser.add_argument("--cam1", help="right DAVIS sensor grayscale frames",
                            default="%s/Baseline/%s/data/right_images" % (dataset_path,
                                dataset_name))
        parser.add_argument("--cam0ts", help="left DAVIS sensor grayscale frames timestamps us",
                            default="%s/Baseline/%s/data/left_images/image_timestamps_left.txt" % (dataset_path,
                                dataset_name))
        parser.add_argument("--cam1ts", help="right DAVIS sensor grayscale frames timestamps us",
                            default="%s/Baseline/%s/data/right_images/image_timestamps_right.txt" % (dataset_path,
                                dataset_name))
        parser.add_argument("--cam0exp", help="left DAVIS sensor grayscale frames exposure us",
                            default="%s/Baseline/%s/data/left_images/image_exposures_left.txt" % (dataset_path,
                                dataset_name))
        parser.add_argument("--cam1exp", help="right DAVIS sensor grayscale frames exposure us",
                            default="%s/Baseline/%s/data/right_images/image_exposures_right.txt" % (dataset_path,
                                dataset_name))
        parser.add_argument("--event_file1", help="Events in 'events' are of type (x, y, t, p)"
                                                  "ms_to_idx is same as in DSEC file format, which allows accessing event chunks using time index",
                            default="%s/Baseline/%s/%s-events_left.h5" % (dataset_path,
                                dataset_name, dataset_name))
        parser.add_argument("--event_file2", help="Events in 'events' are of type (x, y, t, p)"
                                                  "ms_to_idx is same as in DSEC file format, which allows accessing event chunks using time index",
                            default="%s/Baseline/%s/%s-events_right.h5" % (dataset_path,
                                dataset_name, dataset_name))
        args = parser.parse_args()

        cam0_times_file = Path(args.cam0ts)  # In micro seconds NOT nano !!
        cam0_times = np.loadtxt(cam0_times_file)
        cam1_times_file = Path(args.cam1ts)
        cam1_times = np.loadtxt(cam1_times_file)
        cam0_exp_file = Path(args.cam0exp)
        cam0_exp = np.loadtxt(cam0_exp_file)
        #cam0_times += (0.5 * cam0_exp)
        cam1_exp_file = Path(args.cam1exp)
        cam1_exp = np.loadtxt(cam1_exp_file)
        #cam1_times += (0.5 * cam1_exp)
        cam0_images = load_images_from_folder(glob.glob(os.path.join(Path(args.cam0), '*.jpg')))
        cam1_images = load_images_from_folder(glob.glob(os.path.join(Path(args.cam1), '*.jpg')))
        rawfileL = Path(args.event_file1)  # TUM-VIE  1, 2
        rawfileR = Path(args.event_file2)  # TUM-VIE  1, 2
        h, w = 180, 240  # TUM-VIE DVS 720, 1280
        h_c, w_c = 180, 240  # TUM-VIE APS 1024, 1024

        freq_cam = 1 / np.mean(abs(cam0_times[:-1] - cam0_times[1:]) * 1e-6)
        dt_ms = 1000 / freq_cam  # in msec (Events Accumulation Time calculated to correspond to Global Shutter frames)
        delta = 30  # Time Surface Exponential Decay dT (30 msec is good)

        # DAVIS stereo rectification
        # DVS
        kcam_L = np.array([[196.63936292910697, 0, 105.06412666477927], [0, 196.7329768429481, 72.47170071387173], [0, 0, 1]])
        pcam_l = np.array([-0.3367326394292646, 0.11178850939644308, -0.0014005281258491276, -0.00045959441440687044])
        kcam_R = np.array([[196.42564072599785, 0, 110.74517642512458], [0, 196.56440793223533, 88.11310058123058], [0, 0, 1]])
        pcam_R = np.array([-0.3462937629552321, 0.12772002965572962, -0.00027205054024332645, -0.00019580078540073353])
        T_LR = np.array([[0.9991089760393723, -0.04098010198963204, 0.010093821797214667, -0.1479883582369969]
                        , [0.04098846609277917, 0.9991594254283246, -0.000623077121092687, -0.003289908601915284]
                        , [-0.010059803423311134, 0.0010362522169301642, 0.9999488619606629, 0.0026798262366239016]
                        , [0.0, 0.0, 0.0, 1.0]])
        T_RL = np.linalg.inv(T_LR)
        R = T_RL[:3, :3]
        T = T_RL[:3, -1]
        R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(kcam_L, pcam_l, kcam_R, pcam_R, (w, h), R, T)
        xmap1, ymap1 = cv2.initUndistortRectifyMap(kcam_L, pcam_l, R1, kcam_L, (w, h), cv2.CV_32FC1)
        xmap2, ymap2 = cv2.initUndistortRectifyMap(kcam_R, pcam_R, R2, kcam_R, (w, h), cv2.CV_32FC1)
        Ke_L_inv = np.linalg.inv(kcam_L)
        Ke_L = kcam_L
        Ke_R_inv = np.linalg.inv(kcam_R)
        # APS = DVS
        R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(kcam_L, pcam_l, kcam_R, pcam_R, (w_c, h_c), R, T)
        xmap3, ymap3 = cv2.initUndistortRectifyMap(kcam_L, pcam_l, R1, kcam_L, (w_c, h_c), cv2.CV_32FC1)
        xmap4, ymap4 = cv2.initUndistortRectifyMap(kcam_R, pcam_R, R2, kcam_R, (w_c, h_c), cv2.CV_32FC1)
        Kc_L = kcam_L
        Kc_R = kcam_R

        Tce_L = np.eye(4)
        Tce_R = np.eye(4)

        params = ParamsTUMVIE(scale, args.feature)

        h5file = h5py.File(rawfileL)
        print(h5file.keys())
        eventsl = h5file['events']
        print("Left Contains %d events" % (eventsl['t'].shape[0]))
        print("Left Event duration is %.7f seconds" % ((eventsl['t'][-1] - eventsl['t'][0]) * 1e-6))
        eventsL = EventSlicer(h5file)

        h5file = h5py.File(rawfileR)
        print(h5file.keys())
        eventsr = h5file['events']
        print("Right Contains %d events" % (eventsr['t'].shape[0]))
        print("Right Event duration is %.7f seconds" % ((eventsr['t'][-1] - eventsr['t'][0]) * 1e-6))
        eventsR = EventSlicer(h5file)

        delta_uv_L = (+0, +0)
        delta_uv_R = (+0, +0)

    if dataset_type == 'vector':
        parser.add_argument("--cam0", help="left DAVIS sensor grayscale frames",
                            default="%s/VECtor/%s/%s.synced.left_camera" % (dataset_path,
                                dataset_name, dataset_name))
        parser.add_argument("--cam1", help="right DAVIS sensor grayscale frames",
                            default="%s/VECtor/%s/%s.synced.right_camera" % (dataset_path,
                                dataset_name, dataset_name))
        parser.add_argument("--cam0exp", help="left DAVIS sensor grayscale frames exposure us",
                            default="%s/VECtor/%s/%s.synced.left_camera/timestamp.txt" % (dataset_path,
                                dataset_name, dataset_name))
        parser.add_argument("--cam1exp", help="right DAVIS sensor grayscale frames exposure us",
                            default="%s/VECtor/%s/%s.synced.right_camera/timestamp.txt" % (dataset_path,
                                dataset_name, dataset_name))
        parser.add_argument("--event_file1", help="Events in 'events' are of type (x, y, t, p)"
                                                  "ms_to_idx is same as in DSEC file format, which allows accessing event chunks using time index",
                            default="%s/VECtor/%s/%s.synced.left_event.hdf5" % (dataset_path,
                                dataset_name, dataset_name))
        parser.add_argument("--event_file2", help="Events in 'events' are of type (x, y, t, p)"
                                                  "ms_to_idx is same as in DSEC file format, which allows accessing event chunks using time index",
                            default="%s/VECtor/%s/%s.synced.right_event.hdf5" % (dataset_path,
                                dataset_name, dataset_name))
        parser.add_argument("--gt_poses", help="Events in 'events' are of type (x, y, t, p)"
                                               "ms_to_idx is same as in DSEC file format, which allows accessing event chunks using time index",
                            default="%s/VECtor/%s/%s.synced.gt.txt" % (dataset_path,
                                dataset_name, dataset_name))
        args = parser.parse_args()

        gt_poses_file = Path(args.gt_poses)
        gt_poses = np.loadtxt(gt_poses_file)
        cam0_exp_file = Path(args.cam0exp)
        cam0_exp = np.loadtxt(cam0_exp_file)   # Times in seconds so *1e6 to convert to micro
        cam0_times = cam0_exp[:, 1]*1e6
        cam1_exp_file = Path(args.cam1exp)
        cam1_exp = np.loadtxt(cam1_exp_file)
        cam1_times = cam1_exp[:, 1]*1e6
        cam0_images = load_images_from_folder(glob.glob(os.path.join(Path(args.cam0), '*.png')))
        cam1_images = load_images_from_folder(glob.glob(os.path.join(Path(args.cam1), '*.png')))
        rawfileL = Path(args.event_file1)  # TUM-VIE  1, 2
        rawfileR = Path(args.event_file2)  # TUM-VIE  1, 2
        h, w = 480, 640  # TUM-VIE DVS 720, 1280
        h_c, w_c = 1024, 1224  # TUM-VIE APS 1024, 1024

        freq_cam = 1 / np.mean(abs(cam0_times[:-1] - cam0_times[1:]) * 1e-6)
        dt_ms = 1000 / freq_cam  # in msec (Events Accumulation Time calculated to correspond to Global Shutter frames)
        delta = 30  # Time Surface Exponential Decay dT (30 msec is good)

        # DVS stereo rectification
        kcam_L = np.array(
            [[327.32749, 0., 304.97749],
             [0., 327.46184, 235.37621],
             [0., 0., 1.]])
        pcam_l = np.array([-0.031982, 0.041966, -0.000507, -0.001031, 0.000000])
        kcam_R = np.array(
            [[327.48497, 0., 318.53477],
             [0., 327.55395, 230.96356],
             [0., 0., 1.]])
        pcam_R = np.array([-0.026300, 0.037995, -0.000513, 0.000167, 0.000000])
        T_C0_D0 = np.array([[0.9998732356434525, 0.01166113698213495, -0.01084114976267556, -0.0007543180009142757],
                            [-0.01183095928451621, 0.9998062047518974, -0.01573471772912168, -0.04067615384902421],
                            [0.01065556410055307, 0.01586098432919285, 0.9998174273985267, -0.01466127320771003],
                            [0, 0, 0, 1]])
        T_C0_D1 = np.array([[0.9999556435596104, -0.008932688408805093, 0.0029863005670183, 0.1709356441270855],
                            [0.008975438668209301, 0.9998528019716528, -0.0146224447390995, -0.04063388858121166],
                            [-0.002855243246833436, 0.01464859949708519, 0.9998886268573992, -0.01878612060866672],
                            [0, 0, 0, 1]])
        T_LR = np.linalg.inv(T_C0_D0) @ T_C0_D1
        T_RL = np.linalg.inv(T_LR)
        R = T_RL[:3, :3]
        T = T_RL[:3, -1]
        Te = T
        R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(kcam_L, pcam_l, kcam_R, pcam_R, (w, h), R, T)
        xmap1, ymap1 = cv2.initUndistortRectifyMap(kcam_L, pcam_l, R1, kcam_L, (w, h), cv2.CV_32FC1)
        xmap2, ymap2 = cv2.initUndistortRectifyMap(kcam_R, pcam_R, R2, kcam_R, (w, h), cv2.CV_32FC1)
        Ke_L_inv = np.linalg.inv(kcam_L)
        Ke_L = kcam_L
        Ke_R_inv = np.linalg.inv(kcam_R)

        # APS stereo rectification
        kcam_L = np.array(
            [[886.19107, 0., 610.57891],
             [0., 886.59163, 514.59271],
             [0., 0., 1.]])
        pcam_l = np.array([-0.315760, 0.104955, 0.000320, -0.000156, 0.000000])
        kcam_R = np.array(
            [[887.80428, 0., 616.17757],
             [0., 888.04815, 514.71295],
             [0., 0., 1.]])
        pcam_R = np.array([-0.311523, 0.096410, 0.000623, -0.000375, 0.000000])
        T_LR = np.array([[0.9997191305673184, -0.02252628900139945, 0.007363849642027895, 0.1714036578401799],
                         [0.02251384278277662, 0.9997449667048566, 0.001768737249467463, 0.0003819750172057945],
                         [-0.007401814701637792, -0.001602451912123021, 0.9999713222322889, -0.004859483606878026],
                         [0, 0, 0, 1]])
        T_RL = np.linalg.inv(T_LR)
        R = T_RL[:3, :3]
        T = T_RL[:3, -1]
        Tcam = T
        R1, R2, P1, P2, Q, validRoi1, validRoi2 = cv2.stereoRectify(kcam_L, pcam_l, kcam_R, pcam_R, (w_c, h_c), R, T)
        xmap3, ymap3 = cv2.initUndistortRectifyMap(kcam_L, pcam_l, R1, kcam_L, (w_c, h_c), cv2.CV_32FC1)
        xmap4, ymap4 = cv2.initUndistortRectifyMap(kcam_R, pcam_R, R2, kcam_R, (w_c, h_c), cv2.CV_32FC1)
        Kc_L = kcam_L
        Kc_R = kcam_R

        Tce_L = T_C0_D0
        Tce_R = T_RL @ T_C0_D1

        params = ParamsVECtor(scale, args.feature)

        h5file = h5py.File(rawfileL)
        print(h5file.keys())
        eventsl = h5file['events']
        print("Left Contains %d events" % (eventsl['t'].shape[0]))
        print("Left Event duration is %.7f seconds" % ((eventsl['t'][-1] - eventsl['t'][0]) * 1e-6))
        eventsL = EventSlicer(h5file)

        h5file = h5py.File(rawfileR)
        print(h5file.keys())
        eventsr = h5file['events']
        print("Right Contains %d events" % (eventsr['t'].shape[0]))
        print("Right Event duration is %.7f seconds" % ((eventsr['t'][-1] - eventsr['t'][0]) * 1e-6))
        eventsR = EventSlicer(h5file)

        delta_uv_L = (-160, -235)
        delta_uv_R = (-160, -235)

    return Q, params, cam0_images, cam0_times, cam1_images, cam1_times, eventsL, eventsR, dt_ms, h, w, delta, xmap1, ymap1, xmap2, ymap2, xmap3, ymap3, xmap4, ymap4, Kc_L, Tce_L, Ke_L_inv, Kc_R, Tce_R, Ke_R_inv, h_c, w_c, delta_uv_L, delta_uv_R, args
