import numpy as np

from backend import *

dataset_name = "1d"
join = 0  # 1 in case of VECtor

if join:
    save = Path("/media/abanob/DATA/Abanob_PhD/Algorithms_Dev/Python_Alg_RGB-D-T/HCALIB/DH_PTAM/results/save/vec/%s" % (dataset_name))
else:
    save = Path("/media/abanob/DATA/Abanob_PhD/Algorithms_Dev/Python_Alg_RGB-D-T/HCALIB/DH_PTAM/results/save/tum//%s" % (dataset_name))

# Join Trajectory increments for Large-Scale sequences
if join:
    os.chdir(save)
    new_file = open('/%s/DH_PTAM_GPU.txt' % (save), "a")
    new_file.write('#timestamp tx ty tz qx qy qz  qw\n')
    init_pose = np.eye(4)
    curr_update_T = np.eye(4)
    init_time = 0
    for file in sorted(glob.glob("*_j_GPU.txt")):
        curr_traj = np.loadtxt(file)
        for pose_curr in curr_traj:
            if float(pose_curr[0]) > init_time:
                pose_curr_T = np.vstack((np.hstack([Rotation.from_quat(pose_curr[4:]).as_matrix(), pose_curr[1:4].reshape((3,1))]), [0, 0, 0, 1]))
                curr_update_T = init_pose @ pose_curr_T
                curr_update = np.array(np.hstack([curr_update_T[:3, -1].T, Rotation.from_matrix(curr_update_T[:3, :3].reshape((3, 3))).as_quat()]))
                new_file.write(
                    str(pose_curr[0]) + ' ' + str(curr_update[0]) + ' ' + str(curr_update[1]) + ' ' + str(
                        curr_update[2]) + ' ' + str(
                        curr_update[3]) + ' ' + str(curr_update[4]) + ' ' + str(curr_update[5]) + ' ' + str(
                        curr_update[6]) + '\n')
        init_pose = curr_update_T
        init_time = float(curr_traj[-1, 0])
    new_file.close()

est_file = Path("/%s/DH_PTAM_GPU.txt" % (save))
est_traj = np.loadtxt(est_file)

if join:
    gt_file = Path("/%s/gt.txt" % (save))
    gt_traj = np.loadtxt(gt_file)
else:
    gt_file = Path("/%s/mocap_data.txt" % (save))
    gt_traj = np.loadtxt(gt_file)

# Modify the Position output
if join:
    est_traj[:, 0] *= 1e-6
est_traj[:, 1:4] -= est_traj[0, 1:4]

# est_traj[:, [1,2]] = -est_traj[:, [2,1]]
# est_traj[:, [1,3]] = est_traj[:, [3,1]]
# est_traj[:, [2,3]] = est_traj[:, [3,2]]
# est_traj[:, [1,2]] *= -1
# est_traj[:, [3]] *= 0
#
# # Modify the Quaternion output (need to be normalized)
# #est_traj[:, 4:] = np.array([quat_norm(qi) for qi in est_traj[:, 4:]])
# est_traj[:, [5,6]] = est_traj[:, [6,5]]
# est_traj[:, [5,4]] = est_traj[:, [4,5]]
# est_traj[:, [5,6]] *= -1
# #est_traj[:, [4,7]] = est_traj[:, [7,4]]
# est_traj[:, [7]] -= est_traj[0, [7]]

# Align with GT
idx = find_nearest_idx(gt_traj[:, 0], est_traj[0, 0])
est_traj[:, 1:] += gt_traj[idx, 1:]

# Smoothen all estimations
win = 31
n = 3
est_traj[:, 1] = scipy.signal.savgol_filter(est_traj[:, 1], win, n)   # window size, polynomial
est_traj[:, 2] = scipy.signal.savgol_filter(est_traj[:, 2], win, n)   # window size, polynomial
est_traj[:, 3] = scipy.signal.savgol_filter(est_traj[:, 3], win, n)   # window size, polynomial
est_traj[:, 4] = scipy.signal.savgol_filter(est_traj[:, 4], win, n)   # window size, polynomial
est_traj[:, 5] = scipy.signal.savgol_filter(est_traj[:, 5], win, n)   # window size, polynomial
est_traj[:, 6] = scipy.signal.savgol_filter(est_traj[:, 6], win, n)   # window size, polynomial
est_traj[:, 7] = scipy.signal.savgol_filter(est_traj[:, 7], win, n)   # window size, polynomial

# Save the post-processed file
save_file = open('/%s/DH_PTAM_GPU-mod.txt' % (save), "a")
save_file.write('#timestamp tx ty tz qx qy qz  qw\n')
np.savetxt(save_file, est_traj)
save_file.close()

# Plot States
fig0 = mpplot.figure()
# Plot position
ax = fig0.add_subplot(4,2,1)
ax.plot(gt_traj[:,0],gt_traj[:,1],'--',color="red", label="GT")
ax.plot(est_traj[:,0],est_traj[:,1],'-',color="red", label="EST")
ax.set_title('DH-PTAM - Positions')
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_ylabel('X [m]')
ax = fig0.add_subplot(4,2,3)
ax.plot(gt_traj[:,0],gt_traj[:,2],'--',color="red", label="GT")
ax.plot(est_traj[:,0],est_traj[:,2],'-',color="red", label="EST")
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_ylabel('Y [m]')
ax = fig0.add_subplot(4,2,5)
ax.plot(gt_traj[:,0],gt_traj[:,3],'--',color="red", label="GT")
ax.plot(est_traj[:,0],est_traj[:,3],'-',color="red", label="EST")
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_ylabel('Z [m]')
ax.set_xlabel('Time [nsec]')
ax = fig0.add_subplot(4,2,7)
ax.plot(gt_traj[:,1],gt_traj[:,2],'--',color="red", label="GT")
ax.plot(est_traj[:,1],est_traj[:,2],'-',color="red", label="EST")
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_ylabel('Y [m]')
ax.set_xlabel('X [m]')

# Plot quaternion
ax = fig0.add_subplot(4,2,2)
ax.plot(gt_traj[:,0],gt_traj[:,4],'--',color="red", label="GT")
ax.plot(est_traj[:,0],est_traj[:,4],'-',color="red", label="EST")
ax.set_title('DH-PTAM - Quaternions')
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_ylabel('qx [-]')
ax = fig0.add_subplot(4,2,4)
ax.plot(gt_traj[:,0],gt_traj[:,5],'--',color="red", label="GT")
ax.plot(est_traj[:,0],est_traj[:,5],'-',color="red", label="EST")
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_ylabel('qy [-]')
ax = fig0.add_subplot(4,2,6)
ax.plot(gt_traj[:,0],gt_traj[:,6],'--',color="red", label="GT")
ax.plot(est_traj[:,0],est_traj[:,6],'-',color="red", label="EST")
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_ylabel('qz [-]')
ax = fig0.add_subplot(4,2,8)
ax.plot(gt_traj[:,0],gt_traj[:,7],'--',color="red", label="GT")
ax.plot(est_traj[:,0],est_traj[:,7],'-',color="red", label="EST")
ax.grid(color='k', linestyle='--', linewidth=0.25)
ax.set_ylabel('qw [-]')
ax.set_xlabel('Time [nsec]')

ax.legend(["GT", "EST"], loc ="lower left")
fig0.savefig(Path("/%s/DH-PTAM.pdf" % (save)))
mpplot.show()