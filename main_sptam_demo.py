from backend import *

if __name__ == '__main__':
    
    dataset_name = "mocap-desk2"

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-viz', action='store_true', help='do not visualize')
    parser.add_argument('--dataset', type=str, help='dataset (KITTI/EuRoC)', 
        default='tumvie')
    parser.add_argument('--path', type=str, help='dataset path', 
        default='/media/abanobsoliman/My Passport1/Dense_mapping/3Ms/TUM-VIE/%s' %dataset_name)    #/media/abanobsoliman/My Passport1/Dense_mapping/3Ms/TUM-VIE
    args = parser.parse_args()

    if args.dataset.lower() == 'kitti':
        params = ParamsKITTI()
        dataset = KITTIOdometry(args.path)
    elif args.dataset.lower() == 'euroc':
        params = ParamsEuroc()
        dataset = EuRoCDataset(args.path)
    elif args.dataset.lower() == 'tumvie':
        params = ParamsTUMVIE()
        dataset = TUMVIEDataset(args.path)

    sptam = SPTAM(params)

    visualize = not args.no_viz
    if visualize:
        from viewer import MapViewer
        viewer = MapViewer(sptam, params)


    cam = Camera(
        dataset.cam.fx, dataset.cam.fy, dataset.cam.cx, dataset.cam.cy, 
        dataset.cam.width, dataset.cam.height, 
        params.frustum_near, params.frustum_far, 
        dataset.cam.baseline)

    poses_slam_file = open('./ES_PTAM_%s_%s.txt' % (dataset_name, datetime.now().strftime("%d_%m_%Y_%H_%M_%S")), "a")
    poses_slam_file.write('#timestamp tx ty tz qx qy qz  qw\n')

    durations = []
    for i in range(len(dataset)):
        featurel = ImageFeature(dataset.left[i], params)
        featurer = ImageFeature(dataset.right[i], params)
        timestamp = dataset.timestamps[i]

        time_start = time.time()  

        t = Thread(target=featurer.extract)
        t.start()
        featurel.extract()
        t.join()
        
        frame = StereoFrame(i, g2o.Isometry3d(), featurel, featurer, cam, timestamp=timestamp)

        if not sptam.is_initialized():
            sptam.initialize(frame)
        else:
            sptam.track(frame)


        duration = time.time() - time_start
        durations.append(duration)
        print('duration', duration)
        print()
        print()

        SLAM_T_curr = sptam.current.pose.matrix()
        SLAM_pose = np.array(np.hstack([SLAM_T_curr[:3, -1].T, Rotation.from_matrix(SLAM_T_curr[:3, :3].reshape((3, 3))).as_quat()]))
        poses_slam_file.write(str(timestamp) + ' ' + str(SLAM_pose[0]) + ' ' + str(SLAM_pose[1]) + ' ' + str(SLAM_pose[2]) + ' ' + str(SLAM_pose[3]) + ' ' + str(SLAM_pose[4]) + ' ' + str(SLAM_pose[5]) + ' ' + str(SLAM_pose[6]) + '\n')

        if visualize:
            viewer.update()

    print('num frames', len(durations))
    print('num keyframes', len(sptam.graph.keyframes()))
    print('average time', np.mean(durations))

    sptam.stop()
    poses_slam_file.close()
    if visualize:
        viewer.stop()
