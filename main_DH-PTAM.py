import numpy as np

from backend import *

# Memory Cleaner
import gc
import torch
torch.cuda.empty_cache()
gc.collect()

if __name__ == '__main__':

    # Load Dataset
    params, cam0_images, cam0_times, cam1_images, cam1_times, eventsL, eventsR, dt_ms, h, w, delta, xmap1, ymap1, xmap2, ymap2, xmap3, ymap3, xmap4, ymap4, Kc_L, Tce_L, Ke_L_inv, Kc_R, Tce_R, Ke_R_inv, cam, dvs, h_c, w_c, delta_uv_L, delta_uv_R, args = Dataset_loading()
    last_step = len(cam0_images) - 1

    # Current pose
    poses_slam_file = open(
        './results/DH_PTAM_%s_%s.txt' % (args.dataset_name, datetime.now().strftime("%d_%m_%Y_%H_%M_%S")),
        "a")
    poses_slam_file.write('#timestamp tx ty tz qx qy qz  qw\n')
    # keyframes (LC)
    kfs_slam_file = open(
        './results/KFs_DH_PTAM_%s_%s.txt' % (args.dataset_name, datetime.now().strftime("%d_%m_%Y_%H_%M_%S")),
        "a")
    kfs_slam_file.write('#timestamp tx ty tz qx qy qz  qw\n')

    # SuperFeature = match_superglue.SuperGlueMatcher()
    sptam = SPTAM(params)

    if args.visualize:
        from viewer import MapViewer

        viewer = MapViewer(sptam, params)

    kfi = 0
    durations = []
    kfs_times = []
    n_kfs = 0
    # for evsL, evsR in tqdm(zip(EventReader(rawfileL, dt_ms), EventReader(rawfileR, dt_ms)),
    #                        total=len(EventReader(rawfileL, dt_ms))):
    # idx_L = find_nearest_idx(cam0_times, evsL['t'][-1])
    # idx_R = find_nearest_idx(cam1_times, evsR['t'][-1])
    data = []
    for i in tqdm(range(args.skip, last_step)):
        print()

        time_start = time.time()
        # ts0 = time.time()

        timestamp = cam0_times[i]

        if args.dataset_type == "mvsec":
            evsL = slice_events_dict(eventsL, timestamp - dt_ms * 1e3, timestamp)
            evsR = slice_events_dict(eventsR, cam1_times[i] - dt_ms * 1e3, cam1_times[i])
        else:
            evsL = eventsL.get_events(timestamp - dt_ms * 1e3, timestamp)
            evsR = eventsR.get_events(cam1_times[i] - dt_ms * 1e3, cam1_times[i])

        # Load and rectify cam0,1 images
        if args.dataset_type == "mvsec":
            aps_left = cv2.remap(np.array(cam0_images[i]).astype(np.uint8), xmap3, ymap3, cv2.INTER_CUBIC,
                                 borderValue=(255, 255, 255, 255))
            aps_right = cv2.remap(np.array(cam1_images[i]).astype(np.uint8), xmap4, ymap4, cv2.INTER_CUBIC,
                                  borderValue=(255, 255, 255, 255))
        else:
            aps_left = cv2.remap(cv2.imread(cam0_images[i], 0), xmap3, ymap3, cv2.INTER_CUBIC,
                                 borderValue=(255, 255, 255, 255))
            aps_right = cv2.remap(cv2.imread(cam1_images[i], 0), xmap4, ymap4, cv2.INTER_CUBIC,
                                  borderValue=(255, 255, 255, 255))

        #cv2.imshow('APS Frame', cv2.resize(cv2.hconcat((aps_left.astype(np.uint8), aps_right.astype(np.uint8))), (w, int(h/2))))
        #cv2.waitKey(5)

        # # Convert it to pandas DataFrame
        # evsL = pd.DataFrame(evsL)
        # evsR = pd.DataFrame(evsR)
        # # create a sample of your data
        # sample_size = 50000  # adjust this value based on your needs
        # evsL = evsL.sample(n=sample_size)
        # evsR = evsR.sample(n=sample_size)

        ## Draw the spatio-temporal sync
        ## create the 3D plot
        #fig = go.Figure()
        ## add trace
        #fig.add_trace(
        #    go.Scatter3d(
        #        x=evsL['t'], y=evsL['x'], z=evsL['y'],
        #        mode='markers',
        #        marker=dict(
        #            size=0.25,  # reduce size for smaller markers
        #            symbol='circle',
        #            color=['red' if p == 0 else 'blue' for p in evsL['p']],  # assigning colors based on polarity
        #            opacity=0.8
        #        )
        #    )
        #)
        ## adjust the 'camera' settings for desired orientation
        #camera = dict(
        #    up=dict(x=0, y=0, z=1),
        #    center=dict(x=0, y=0, z=0),
        #    eye=dict(x=1.25, y=1.25, z=1.25)
        #)
        #fig.update_layout(scene=dict(xaxis_title='t', yaxis_title='x', zaxis_title='y', camera=camera, aspectratio=dict(x=.2, y=1, z=1)))
        #fig.show()
        #breakpoint()

        # EventReader object for reading chunk-by-chunk
        evL_arr = np.stack([evsL['x'], evsL['y'], evsL['t'], evsL['p']], axis=1)
        evR_arr = np.stack([evsR['x'], evsR['y'], evsR['t'], evsR['p']], axis=1)

        # Event 3-Channel Tensors Creation
        imgL = e3ct_create(dt_ms, evL_arr, h, w, delta, xmap1, ymap1, aps_left, Kc_L, Tce_L, Ke_L_inv, delta_uv_L)
        imgR = e3ct_create(dt_ms, evR_arr, h, w, delta, xmap2, ymap2, aps_right, Kc_R, Tce_R, Ke_R_inv, delta_uv_R)

        # print('Construct', time.time() - ts0)
        # ts1 = time.time()

        if np.mean(aps_left) <= 50.0 or np.mean(aps_left) >= 200.0:
            alpha_left = np.min([np.max([np.mean(aps_left) / np.max(aps_left), 1.0 - np.mean(aps_left) / np.max(aps_left)]), args.beta_lim])
            print("Left Fusion frame - DVS biased, with beta = ", alpha_left)
        else:
            alpha_left = np.max([np.min([np.mean(aps_left) / np.max(aps_left), 1.0 - np.mean(aps_left) / np.max(aps_left)]), args.beta_lim])
            print("Left Fusion frame - APS biased, with beta = ", alpha_left)

        if np.mean(aps_right) <= 50.0 or np.mean(aps_right) >= 200.0:
            alpha_right = np.min([np.max([np.mean(aps_right) / np.max(aps_right), 1.0 - np.mean(aps_right) / np.max(aps_right)]), args.beta_lim])
            print("Right Fusion frame - DVS biased, with beta = ", alpha_right)
        else:
            alpha_right = np.max([np.min([np.mean(aps_right) / np.max(aps_right), 1.0 - np.mean(aps_right) / np.max(aps_right)]), args.beta_lim])
            print("Right Fusion frame - APS biased, with beta = ", alpha_right)

        data.append([alpha_left, np.mean(aps_left) / np.max(aps_left), alpha_right, np.mean(aps_right) / np.max(aps_right)])
        #if i >= 70:
        #   break
        #continue

        ti = Thread(target=imgL.fuse(alpha_left))
        ti.start()
        imgR.fuse(alpha_right)
        ti.join()

        sensor = cam
        imageL = aps_left  # imgL.fusion   #imgL.fusion  # Fusion: imgL.E3CT + aps_left
        imageR = aps_right  #imgR.fusion  #imgR.fusion  # Fusion: imgR.E3CT + aps_right

        # print('Fuse', time.time() - ts1)
        # ts2 = time.time()

        #cv2.imshow('E3CT_rect Frame',
        #           cv2.resize(cv2.hconcat((imgL.fusion.astype(np.uint8), imgR.fusion.astype(np.uint8))), (aps_left.shape[1], int(aps_left.shape[0]/2))))
        #cv2.imshow('E3CT Frame',
        #           cv2.resize(cv2.hconcat((imgL.E3CT.astype(np.uint8), imgR.E3CT.astype(np.uint8))), (w, int(h/2))))
        #cv2.waitKey(5)
        #continue

        featurel = ImageFeature(imageL, params)  # Select: imgL.E3CT or aps_left
        featurer = ImageFeature(imageR, params)  # Select: imgR.E3CT or aps_right

        t = Thread(target=featurel.extract())
        t.start()
        featurer.extract()
        t.join()

        frame = StereoFrame(kfi, g2o.Isometry3d(), featurel, featurer, sensor, timestamp=timestamp)

        # print('Detect', time.time() - ts2)

        if not sptam.is_initialized():
            #ts3 = time.time()
            sptam.initialize(frame)
            #print('Initialize', time.time() - ts3)
        else:
            sptam.track(frame)

        duration = time.time() - time_start
        durations.append(duration)
        print('duration', duration)
        print()

        SLAM_T_curr = sptam.current.pose.matrix()
        SLAM_pose = np.array(
            np.hstack([SLAM_T_curr[:3, -1].T, Rotation.from_matrix(SLAM_T_curr[:3, :3].reshape((3, 3))).as_quat()]))
        poses_slam_file.write(
            str(timestamp) + ' ' + str(SLAM_pose[0]) + ' ' + str(SLAM_pose[1]) + ' ' + str(SLAM_pose[2]) + ' ' + str(
                SLAM_pose[3]) + ' ' + str(SLAM_pose[4]) + ' ' + str(SLAM_pose[5]) + ' ' + str(SLAM_pose[6]) + '\n')

        # # Draw Circular Match
        # if len(sptam.graph.keyframes()) >= 2:
        #     prev_ptsL = []
        #     prev_ptsR = []
        #     curr_ptsL = []
        #     curr_ptsR = []
        #     kf = sptam.graph.keyframes()
        #     for m in kf[-2].measurements():
        #         if len(cv2.KeyPoint_convert(m.get_keypoints())) > 1:
        #             prev_ptsL.append((cv2.KeyPoint_convert(m.get_keypoints())[0]).astype(int).tolist())
        #             prev_ptsR.append((cv2.KeyPoint_convert(m.get_keypoints())[1]).astype(int).tolist())
        #     for m in kf[-1].measurements():
        #         if len(cv2.KeyPoint_convert(m.get_keypoints())) > 1:
        #             curr_ptsL.append((cv2.KeyPoint_convert(m.get_keypoints())[0]).astype(int).tolist())
        #             curr_ptsR.append((cv2.KeyPoint_convert(m.get_keypoints())[1]).astype(int).tolist())
        #     circ_img = cv2.cvtColor(cv2.vconcat((cv2.hconcat(
        #         (imgL.fusion.astype(np.uint8), imgR.fusion.astype(np.uint8))),
        #                                          cv2.hconcat((prev_fuse_L, prev_fuse_R)))), cv2.COLOR_GRAY2RGB)
        #     cv2.imshow('SuperPoints Circular Match', cv2.resize(circ_img, (900, 900)))
        #     curr_ptsL = np.array(curr_ptsL)
        #     curr_ptsR = np.array(curr_ptsR)
        #     prev_ptsL = np.array(prev_ptsL)
        #     prev_ptsR = np.array(prev_ptsR)
        #     curr_ptsR += [w_c, 0]
        #     prev_ptsL += [0, h_c]
        #     prev_ptsR += [w_c, h_c]
        #     test = curr_ptsL + [0, h_c]
        #     for l in range(1, 50):
        #         cv2.line(circ_img, (int(curr_ptsL[l, 0]), int(curr_ptsL[l, 1])),
        #                  (int(curr_ptsR[l, 0]), int(curr_ptsR[l, 1])), (0, 255, 0), 3)
        #         cv2.line(circ_img, (int(prev_ptsL[l, 0]), int(prev_ptsL[l, 1])),
        #                  (int(prev_ptsR[l, 0]), int(prev_ptsR[l, 1])), (255, 0, 0), 3)
        #         cv2.line(circ_img, (int(test[l, 0]), int(test[l, 1])),
        #                  (int(curr_ptsL[l, 0]), int(curr_ptsL[l, 1])), (0, 0, 255), 3)
        #     cv2.imshow('SuperPoints Circular Match', cv2.resize(circ_img, (900, 900)))
        #     cv2.imwrite('./results/circ_%s_%s.png' % (args.dataset_name, datetime.now().strftime("%d_%m_%Y_%H_%M_%S")),
        #                 circ_img)
        #     cv2.waitKey(5)
        #     breakpoint()
        # prev_fuse_L = imgL.fusion.astype(np.uint8)
        # prev_fuse_R = imgR.fusion.astype(np.uint8)

        kfi += 1

        if len(sptam.graph.keyframes()) > n_kfs:
            kfs_times.append(timestamp)

        n_kfs = len(sptam.graph.keyframes())

        if args.visualize:
            viewer.update(featurel.draw_keypoints())

    poses_slam_file.close()

    # assuming data is a list of lists
    df = pd.DataFrame(data, columns=['beta_left', 'aps_left', 'beta_right', 'aps_right'])
    # generating frame numbers array (replace this with your actual frame numbers if you have them)
    frames = np.arange(1, len(df) + 1)
    # Create subplot figure
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Left Camera", "Right Camera"))
    # Create a trace for the left camera aps
    trace_left_aps = go.Scatter(
        x=frames,
        y=df['aps_left'],
        mode='lines+markers',
        name=r'${\bar{C}}/{C^{max}}$',
        line=dict(color='blue'),
        marker=dict(size=7)
    )
    # Create a trace for the left camera beta
    trace_left_beta = go.Scatter(
        x=frames,
        y=df['beta_left'],
        mode='lines+markers',
        name=r'$\beta$',
        line=dict(color='green'),
        marker=dict(size=7)
    )
    # Add traces to the subplot
    fig.add_trace(trace_left_aps, row=1, col=1)
    fig.add_trace(trace_left_beta, row=1, col=1)
    # Create a trace for the right camera aps
    trace_right_aps = go.Scatter(
        x=frames,
        y=df['aps_right'],
        mode='lines+markers',
        name=r'${\bar{C}}/{C^{max}}$',
        line=dict(color='red'),
        marker=dict(size=7)
    )
    # Create a trace for the right camera beta
    trace_right_beta = go.Scatter(
        x=frames,
        y=df['beta_right'],
        mode='lines+markers',
        name=r'$\beta$',
        line=dict(color='purple'),
        marker=dict(size=7)
    )
    # Add traces to the subplot
    fig.add_trace(trace_right_aps, row=2, col=1)
    fig.add_trace(trace_right_beta, row=2, col=1)
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Events-Frames Fusion Modes Analysis",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        ),
        annotations=[
            dict(
                x=0.5,
                y=1,
                showarrow=False,
                text="Left Camera",
                xref="paper",
                yref="paper",
                font=dict(
                    family="Courier New, monospace",
                    size=25,
                    color="#7f7f7f"
                ),
            ),
            dict(
                x=0.5,
                y=0.38,
                showarrow=False,
                text="Right Camera",
                xref="paper",
                yref="paper",
                font=dict(
                    family="Courier New, monospace",
                    size=25,
                    color="#7f7f7f"
                ),
            )
        ],
        legend=dict(
            font=dict(
                family="Courier New, monospace",
                size=30,
                color="#7f7f7f"
            )
        )
    )

    fig.update_xaxes(title_text='Frame Number', row=1, col=1)
    fig.update_xaxes(title_text='Frame Number', row=2, col=1)
    fig.update_yaxes(title_text='Metrics (APS, Beta)', row=1, col=1)
    fig.update_yaxes(title_text='Metrics (APS, Beta)', row=2, col=1)
    fig.show()

    print('num frames', len(durations))
    print('num keyframes', len(sptam.graph.keyframes()))
    print('average time', np.mean(durations))

    for kf in sptam.graph.keyframes():
        SLAM_T_curr = kf.pose.matrix()
        SLAM_pose = np.array(
            np.hstack([SLAM_T_curr[:3, -1].T, Rotation.from_matrix(SLAM_T_curr[:3, :3].reshape((3, 3))).as_quat()]))
        kfs_slam_file.write(
            str(kfs_times[kf.id]) + ' ' + str(SLAM_pose[0]) + ' ' + str(SLAM_pose[1]) + ' ' + str(
                SLAM_pose[2]) + ' ' + str(
                SLAM_pose[3]) + ' ' + str(SLAM_pose[4]) + ' ' + str(SLAM_pose[5]) + ' ' + str(SLAM_pose[6]) + '\n')
    kfs_slam_file.close()

    sptam.stop()

    if args.visualize:
        viewer.stop(featurel.draw_keypoints())
