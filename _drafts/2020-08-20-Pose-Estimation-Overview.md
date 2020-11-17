---
layout: post
title: Pose Estimation
date: 2020-10-18 10:23:16 +0200
permalink: /:title
author: Johann Gerberding
---

# (Human) Pose Estimation

- most of the time Human Pose Estimation
- many benchmarks are based on images of humans, maybe because of applications where it can be used
- in this article I will only talk about Deep Learning based Pose Estimation models
- nowadays all Pose Estimation Models consist of a DL part (ConvNet)
- first convolutional neural net was applied in 2014 (Jain et al.)
- before ConvNets the best approaches for this task were based on body part detectors (multiple stages of processing)


What is Pose Estimation?
What is the state of the art in the field of pose estimation

- 2D Pose Estimation -> estimate a 2D pose (x,y) coordinates for each joint from a RGB image
- 3D Pose Estimation -> estimate a 3D pose (x,y,z) coordinates 


## Top-down vs. Bottom-up Approaches

Description of the two different approaches
How does pose estimation work?

## Evaluation

In the follow I will describe the most common evaluation metrics in Human Pose Estimation which are important to compare different approaches.

### Percentage of Correct Parts (PCP)

A limb is considered detected  (a correct part) if the distance between the two predicted joint locations and the true limb joint locations is less than half of the limb length (Commonly denoted as PCP@0.5). The smaller the PCP value, the better the performance of the model. The drawback of this metric is that it penalizes shorter limbs because shorter limbs like lower arms which are usually harder to detect.

### Percentage of Correct Keypoints (PCK)



### Percentage of Detected Joints (PDJ)

PDJ measures the detection rate of joints where a joint is considered as detected if the distance between the predicted joint and the ground-truth joint is less than a fraction of a **base element**.

The base element from the original implementation is the **torso diameter** which is defined as the distance between left shoulder and right hip of each ground truth pose. But there is a problem with this metric if the person in the 2D image turns sideways. In this case the distance between the right and the left shoulder come close to zero. The same goes for the left and the right hips.This reduces the length of the torso diameter. Instead you can use the diagonal of the bounding box as an alternative base element. The formula to calculate the PDJ looks like this:

$$PDJ = \frac{\sum_{i=1}^{n} bool(d_{i} < threshold * diagonal_{bbox})}{n}$$

where $d_{i}$ represents the euclidean distance between the ground truth keypoint and the predicted keypoint. The $bool()$ function returns 1 if the distance is smaller than the diagonal times the threshold (e.g. 0.05) and 0 if its not. The integer $n$ describes the number of keypoints.

### Object Keypoint Similarity (OKS) based mAP

This evaluation metric is used for the Keypoint Evaluation in the COCO benchmark (https://cocodataset.org/#keypoints-eval). For each object in this dataset the ground truth keypoints have the form $[x_{1}, y_{1}, v_{1}, ..., x_{k}, y_{k}, v_{k}]$ where $x$ and $y$ are the keypoint locations and $v$ is a visibility flag (0 => not labeled; 1 => labeled but not visible; 2 => labeled and visible). On top of that each ground truth label has a scale $s$ shich is defined as the square root of the object segment area. 

The OKS is defined as followed:

$$OKS = \frac{\sum_{i} exp(-d_{i}^{2} / 2s^{2}k_{i}^{2}) \delta (v_{i} > 0)}{\sum_{i} \delta (v_{i} > 0)}$$

Here $d_{i}$ also describes the euclidean distance between the ground truth keypoint and the detection and $k_{i}$ is a per-keypoint constant that controls falloff.

## Datasets/Benchmarks

What are the most popular datasets in this domain

- Frames Labeled in Cinema (FLIC) (https://bensapp.github.io/flic-dataset.html)
- MPII Human Pose Models (http://pose.mpi-inf.mpg.de/)
- COCO
- Leeds Sports Pose Dataset (https://sam.johnson.io/research/lsp.html)

## Important Architectures

Description of a few popular architectures in pose estimation

### Learning Human Pose Estimation Features with Convolutional Networks (2014)

- first deep learning approach to tackle the problem of human pose estimation (conv nets)
- architecture:

![model architecture of Jain et al. 2014](../assets/imgs/model_architecture_jain_et_al_2014.png)

- they used multiple convnets to perform independent binary body-part classification
- sliding window approach, output 1 or 0 if body part is in this region
- convnet produces a *response map* indicating the confidence of the body part at that location

- before feeding the image into the network, local contrast normalization is performed

## References

- general information: https://nanonets.com/blog/human-pose-estimation-2d-guide/?utm_source=reddit&utm_medium=social&utm_campaign=pose&utm_content=GROUP_NAME

- cool github with pose estimation papers: https://github.com/cbsudux/awesome-human-pose-estimation

