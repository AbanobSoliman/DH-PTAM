# DH-PTAM

This python research project is the complete implementation of DH-PTAM.  

Insights video:

[![DG-PTAM Insights Video](results/dhptam_vid.png)](https://youtu.be/chAC-vQH9KU)

Our implementation inspires your research ? Are you interested to use our code !
Thank you for citing our paper:

    A. Soliman, F. Bonardi, D. Sidibé, and S. Bouchafa, "DH-PTAM: A Deep Hybrid Stereo Events-Frames Parallel Tracking And Mapping System," 
    IEEE Transactions on Intelligent Vehicles, vol. 0, 2024, doi: 10.1109/TIV.2024.3412595.

## Requirements
* Python 3.6+
* numpy, numba, hdf5plugin, progress, tqdm, skimage, scipy, argparse
* cv2
* [g2o](https://github.com/uoip/g2opy) <sub>(python binding of C++ library [g2o](https://github.com/RainerKuemmerle/g2o))</sub> for optimization
* [pangolin](https://github.com/uoip/pangolin) <sub>(python binding of C++ library [Pangolin](http://github.com/stevenlovegrove/Pangolin))</sub> for visualization

## Usage
* The entry point where we select the dataset and DH-PTAM parameters selection:
`main_DH-PTAM.py`  
* Setting parameters configuration:
`params.py`  
* Setting datasets configuration (Set for MVSEC, TUM-VIE and VECtor):
`dataset_config.py`  

## Results
The trajectories of all experiments reported in the paper are given in:   
* /DH_PTAM/results/save/

## Credits
The backend of this project is based on:
* PySLAM: https://github.com/luigifreda/pyslam
* S-PTAM: https://github.com/uoip/stereo_ptam
