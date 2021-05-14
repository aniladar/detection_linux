# Setup Instructions for the Linux machine to test CNNs
This repo is cloned from [AIGUYS' deep-sort repo](https://github.com/theAIGuysCode/yolov4-deepsort) 
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

\!!! For GPU usage, if Conda Environment (Anaconda) is not used (just for Windows, Linux does not need it), the environment has not set up CUDA yet. Make sure to use CUDA Toolkit version 10.1 . Repo for proper CUDA is
https://developer.nvidia.com/cuda-10.1-download-archive-update2

## Pre-trained Network
The object tracker uses YOLOv4 to detect the objects, which deep-sort then uses to track. In order not to wait for training time in every runtime, official weights can be used using transfer learning.
<br/>
<br/> Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT
<br/>
or<br/>

```bash
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -P data/
```

## Running the Tracker with YOLOv4
To implement the object tracking, we need to convert the .weights into the corresponding TensorFlow model (tf or tflite) which will be saved to a checkpoints folder. Then all we need to do is run the object_tracker.py script to run the object tracker.

```bash
# Convert weights to tensorflow model
python save_model.py --model yolov4 

# Run yolov4 deep sort object tracker on video (to record detection on a video file, set --output flag to related directory)
python object_tracker.py --video ./data/video/test.mp4 --output --model yolov4 --info

# for terminal output (set dont_show flag)
python object_tracker.py --video ./data/video/test.mp4 --output --model yolov4 --dont_show --info

# Run yolov4 deep sort object tracker on live camera (to record detection on a video file, set --output flag to related directory)
python object_tracker.py --video 0 --output --model yolov4 --info
```

## Command Line Args Reference

```bash
save_model.py:
  --weights: path to weights file
    (default: './data/yolov4.weights')
  --output: path to output
    (default: './checkpoints/yolov4-416')
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'False')
  --input_size: define input size of export model
    (default: 416)
  --framework: framework type to use (tf, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
    
 object_tracker.py:
  --video: path to input video (use 0 for videocamera)
    (default: './data/video/test.mp4')
  --output: path to output video 
    (default: None)
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './checkpoints/yolov4-416')
  --framework: framework type to use (tf, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
  --size: resize images to
    (default: 416)
  --iou: iou threshold
    (default: 0.45)
  --score: confidence threshold
    (default: 0.50)
  --dont_show: dont show video output
    (default: False)
  --info: print detailed info about tracked objects
    (default: False)
```
