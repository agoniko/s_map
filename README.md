# Real-Time 3D Semantic Mapping System

This repository contains the code and materials for a **real-time 3D semantic mapping system** developed as part of the Master's thesis: *Real-Time 3D Semantic Mapping with Multimodal Data for Autonomous Robotic Navigation*. The system enables autonomous robots to build detailed semantic maps of their environments by fusing RGB and depth data for enhanced navigation and scene understanding.

## Demo
Click the image below to watch the demo on YouTube:

[![Watch the video](./thesis_material/demo.gif)](https://youtu.be/lMwoXoRP1LY)

## Abstract
Effective navigation in complex environments is crucial for autonomous robots. This system leverages pre-trained deep learning models for object segmentation and integrates depth data to construct 3D semantic maps. Key features include:
- **Real-time processing**: Achieving an average of 11 FPS.
- **Efficient object re-identification**: Utilizing KD-Trees and voxel grids.
- **High mapping precision**: Evaluated with a mean Average Precision (mAP) score of 0.8 at a 40% object overlap threshold.

The system was tested in an indoor environment, demonstrating high accuracy and efficiency for robotic applications.

## Thesis
The complete thesis, with detailed methodology, results, and future directions, can be found [here](./thesis_material/Real_Time_3D_Semantic_Mapping_with_Multimodal_Data_for_Autonomous_Robotic_Navigation.pdf).