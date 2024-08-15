# Object_Detection
# Object Detection with YOLOv2 (608x608)

This project implements object detection using the YOLOv2 (You Only Look Once) model with a 608x608 input image size. YOLOv2 is a real-time object detection system that can identify multiple objects in images or video streams.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Introduction

YOLOv2, also known as YOLO9000, is an advanced version of the original YOLO model that improves detection accuracy and speed. This implementation uses a 608x608 input size to enhance detection performance, especially for smaller objects.

## Prerequisites

Before running this project, ensure you have the following software installed:

- Python 3.x
- TensorFlow 2.x
- OpenCV
- NumPy
- Keras
- Pillow

You can install the required Python packages using `pip`:

# Generating YOLOv2 Model
If you need the yolov2.h5 file, you can generate it as follows:

For anyone who still has this problem in TensorFlow 2, go to the original website of YOLOv2 and download Darknet and weights file. Find the YOLO config in the Darknet folder, then using YAD2K you can easily generate the .h5 file. When running yad2k.py, you may get a bunch of import errors that you have to manually correct due to newer versions of TensorFlow.

YOLOv2 Site:
YOLOv2 Official Website

Installing YAD2K:
git clone https://github.com/allanzelener/yad2k.git
Put the yolo.cfg and yolo.weights files inside the yad2k folder.
If you get a space_to_depth error, go to the corresponding line in yad2k and change it to tf.nn.space_to_depth.
Generating the yolov2.h5 File:

The yolo.h5 file can be generated using the YAD2K repository:

Clone the YAD2K repository:
git clone https://github.com/allanzelener/yad2k.git
Download the YOLO weights file:

Download YOLO Weights

Download the YOLO config file from the YOLO website.

Run the following command to generate the yolo.h5 file:
python yad2k.py yolo.cfg yolo.weights model_data/yolo.h5

# Acknowledgments
## YOLOv2: Developed by Joseph Redmon and others. For more details, visit YOLOv2 GitHub.
## TensorFlow and Keras: Used for building and training the model.
## Andrew Ng








