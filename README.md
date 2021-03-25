# Setup Instructions for the Linux machine to test CNNs
## Requirements
 * Python 3.8
 * Tensorflow 2.3.0
 * openCV-python 4.1.2.30
 * lxml
 * tqdm
 * absl-py
 * easydict
 * matlplotlib
 * pillow
## Getting Started
To get started, install the proper dependencies

```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

\!!! For GPU usage, if Conda Environment is not used, the environment has not set up CUDA yet. Make sure to use CUDA Toolkit version 10.1 . Repo for proper CUDA is
https://developer.nvidia.com/cuda-10.1-download-archive-update2

## Pre-trained Network
The object tracker uses YOLOv4 to detect the objects, which deep-sort then uses to track. In order not to wait for training time every runtime, official weights can be used using transfer learning.
<br/>
<br/> Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT
