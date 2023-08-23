# DH-PTAM

This python research project is the authors complete implementation of DH-PTAM.  

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
* Setting datasets configuration (Set for TUM-VIE and VECtor):
`dataset_config.py`  

## Results
The trajectories of all experiments reported in the paper are given in:   
* /DH_PTAM/results/save/
