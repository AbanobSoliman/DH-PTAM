import h5py
import hdf5plugin
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R


gt_file = h5py.File(Path("/media/abanob/My Passport/Dense_mapping/3Ms/MVSEC/indoor_flying4_gt.hdf5"))
poses = gt_file['davis']['left']['pose'][:]
timestamps = gt_file['davis']['left']['pose_ts'][:]

# Open file for writing
with open('/media/abanob/DATA/DH-PTAM/results/MVSEC/indoor_flying4_gt.txt', 'w') as f:
    # Write header
    f.write('#timestamp tx ty tz qx qy qz qw\n')

    # Process each pose and timestamp
    for timestamp, pose in zip(timestamps, poses):
        tx, ty, tz = pose[:3, 3]
        quat = R.from_matrix(pose[:3, :3]).as_quat()
        f.write(f'{timestamp} {tx} {ty} {tz} {quat[0]} {quat[1]} {quat[2]} {quat[3]}\n')
