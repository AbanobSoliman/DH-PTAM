from backend_FE_exp import *

directory_save = '/media/abanob/My Passport/Dense_mapping/3Ms/Baseline/results/desk'
if not os.path.exists(directory_save):
        os.makedirs(directory_save)

if __name__ == '__main__':

    # Load Dataset
    Q, params, cam0_images, cam0_times, cam1_images, cam1_times, eventsL, eventsR, dt_ms, h, w, delta, xmap1, ymap1, xmap2, ymap2, xmap3, ymap3, xmap4, ymap4, Kc_L, Tce_L, Ke_L_inv, Kc_R, Tce_R, Ke_R_inv, h_c, w_c, delta_uv_L, delta_uv_R, args = Dataset_loading()
    last_step = len(cam0_images) - 1

    kfi = 0
    durations = []
    kfs_times = []
    n_kfs = 0

    for i in tqdm(range(args.skip, last_step)):
        print()

        time_start = time.time()
        # ts0 = time.time()

        timestamp = cam0_times[i]
        evsL = eventsL.get_events(timestamp - dt_ms * 1e3, timestamp)
        evsR = eventsR.get_events(cam1_times[i] - dt_ms * 1e3, cam1_times[i])
        # Load and rectify cam0,1 images
        aps_left = cv2.remap(cv2.imread(cam0_images[i], 0), xmap3, ymap3, cv2.INTER_CUBIC,
                             borderValue=(255, 255, 255, 255))
        aps_right = cv2.remap(cv2.imread(cam1_images[i], 0), xmap4, ymap4, cv2.INTER_CUBIC,
                              borderValue=(255, 255, 255, 255))

        # # Convert it to pandas DataFrame
        # evsL = pd.DataFrame(evsL)
        # evsR = pd.DataFrame(evsR)
        # # create a sample of your data
        # sample_size = 50000  # adjust this value based on your needs
        # evsL = evsL.sample(n=sample_size)
        # evsR = evsR.sample(n=sample_size)

        # EventReader object for reading chunk-by-chunk
        evL_arr = np.stack([evsL['x'], evsL['y'], evsL['t'], evsL['p']], axis=1)
        evR_arr = np.stack([evsR['x'], evsR['y'], evsR['t'], evsR['p']], axis=1)

        # Event 3-Channel Tensors Creation
        imgL = e3ct_create(dt_ms, evL_arr, h, w, delta, xmap1, ymap1, aps_left, Kc_L, Tce_L, Ke_L_inv, delta_uv_L)
        imgR = e3ct_create(dt_ms, evR_arr, h, w, delta, xmap2, ymap2, aps_right, Kc_R, Tce_R, Ke_R_inv, delta_uv_R)

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

        ti = Thread(target=imgL.fuse(alpha_left))
        ti.start()
        imgR.fuse(alpha_right)
        ti.join()

        imageL = imgL.fusion   #imgL.fusion  # Fusion: imgL.E3CT + aps_left
        imageR = imgR.fusion  #imgR.fusion  # Fusion: imgR.E3CT + aps_right

        # Calculate depth maps
        depth_map_e3ct, e3ct_pcl = calc_depth_map(imgL.E3CT, imgR.E3CT, Q, imgL.mask)
        depth_map_fuse, image_pcl = calc_depth_map(imgL.fusion, imgR.fusion, Q, imgL.mask)
        fuse_depth = fuse_images(depth_map_fuse, enhance_intensity(aps_left), alpha=0.9)

        #cv2.imshow('E3CT_rect Frame',
        #           cv2.resize(cv2.hconcat((imgL.fusion.astype(np.uint8), imgR.fusion.astype(np.uint8))), (aps_left.shape[1]*4, int(aps_left.shape[0]*2))))
        #cv2.imshow('E3CT Frame',
        #           cv2.resize(cv2.hconcat((imgL.E3CT.astype(np.uint8), imgR.E3CT.astype(np.uint8))), (w*4, int(h*2))))

        # Plot the depth map (before and after) fusion
        #cv2.imshow('E3CT Depth', cv2.resize(depth_map_e3ct.astype(np.uint8), (int(w * 2), int(h * 2))))
        #cv2.imshow('Depth Map', cv2.resize(depth_map_fuse.astype(np.uint8), (int(w_c * 2), int(h_c * 2))))
        #cv2.imshow('APS Frame', cv2.resize(aps_left.astype(np.uint8), (int(w_c * 2), int(h_c * 2))))
        save_1 = enhance_brightness_saturation(fuse_depth)
        cv2.imwrite(os.path.join(directory_save, f'depth_image_{i + 1}.png'), save_1)
        cv2.imshow('Fusion Depth', cv2.resize(fuse_depth.astype(np.uint8), (int(w_c * 2), int(h_c * 2))))
        save_2 = image_pcl.astype(np.uint8)
        cv2.imwrite(os.path.join(directory_save, f'pcd_image_{i + 1}.png'), save_2)
        cv2.imshow('Fusion PCL', cv2.resize(image_pcl.astype(np.uint8), (int(w_c * 2), int(h_c * 2))))

        cv2.waitKey(5)
