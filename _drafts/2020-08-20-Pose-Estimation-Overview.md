---
layout: post
title: Pose Estimation
date: 2020-10-18 10:23:16 +0200
permalink: /:title
author: Johann Gerberding
---

# (Human) Pose Estimation

When people in the Machine Learning Community talk about Pose Estimation, one can usually assume that they are talking about Human Pose Estimation. All of the known benchmarks in the field of Pose Estimation are based on images of people. Maybe this is due to the many potential applications of such models, e.g. Action Recognition, ... This is a difficult problem due to possible strong articulations, small and barely visible joints or keypoints, occlusions, and a high variance in clothing and lighting.

But first of all, what exactly is Human Pose Estimation?

Basically you can differentiate between 2D and 3D pose estimation. In **2D Pose Estimation** a 2D pose of (x,y) coordinates for each joint from a RGB image are estimated. In **3D Pose Estimation** you also incorporate the prediction of a third coordinate z. In this article I will only talk about Deep Learning based Pose Estimation models because nowadays nearly all Pose Estimation Models consist of a Deep Learning part (Convolutional Neural Network). The first ConvNet was applied in 2014 [3] by Jain et al. Before this the best approaches for this task were based on body part detectors (multiple stages of processing).


## Top-down vs. Bottom-up Approaches

There are two different approaches on how to tackle the problem of Pose Estimation: Top-down and Bottom-up.

**Top-down:** This is a two stage approach where you combine a detection model with a pose estimation model. The detection model first predicts bounding boxes of the people in an image which you feed the pose estimation model which predicts the keypoints for the person. This approach depends on the performance of the upstream detection model and can be computationally expensive.

**Bottom-up:** 

## Evaluation

In the follow I will describe the most common evaluation metrics in Human Pose Estimation which are important to compare different approaches.

### Percentage of Correct Parts (PCP)

A limb is considered detected  (a correct part) if the distance between the two predicted joint locations and the true limb joint locations is less than half of the limb length (Commonly denoted as PCP@0.5). The smaller the PCP value, the better the performance of the model. The drawback of this metric is that it penalizes shorter limbs because shorter limbs like lower arms which are usually harder to detect.

### Percentage of Detected Joints (PDJ)

PDJ measures the detection rate of joints where a joint is considered as detected if the distance between the predicted joint and the ground-truth joint is less than a fraction of a **base element**.

The base element from the original implementation is the **torso diameter** which is defined as the distance between left shoulder and right hip of each ground truth pose. But there is a problem with this metric if the person in the 2D image turns sideways. In this case the distance between the right and the left shoulder come close to zero. The same goes for the left and the right hips.This reduces the length of the torso diameter. Instead you can use the diagonal of the bounding box as an alternative base element. The formula to calculate the PDJ looks like this:

$$PDJ = \frac{\sum_{i=1}^{n} bool(d_{i} < threshold * diagonal_{bbox})}{n}$$

where $d_{i}$ represents the euclidean distance between the ground truth keypoint and the predicted keypoint. The $bool()$ function returns 1 if the distance is smaller than the diagonal times the threshold (e.g. 0.05) and 0 if its not. The integer $n$ describes the number of keypoints.


### Percentage of Correct Keypoints (PCK)

(original paper: Multimodal decomposable models for human pose estimation 2013)

Can be the same as PDJ where the torso diameter is used as base element for the threshold calculation. The threshold can also be 50% of the head bone link, which is denoted as PCKh@0.5 [2]. This alleviates the problem with shorter limbs.


### Object Keypoint Similarity (OKS) based mAP

This evaluation metric is used for the Keypoint Evaluation in the COCO benchmark (https://cocodataset.org/#keypoints-eval). For each object in this dataset the ground truth keypoints have the form $[x_{1}, y_{1}, v_{1}, ..., x_{k}, y_{k}, v_{k}]$ where $x$ and $y$ are the keypoint locations and $v$ is a visibility flag (0 => not labeled; 1 => labeled but not visible; 2 => labeled and visible). On top of that each ground truth label has a scale $s$ shich is defined as the square root of the object segment area. 

The OKS is defined as followed:

$$OKS = \frac{\sum_{i} exp(-d_{i}^{2} / 2s^{2}k_{i}^{2}) \delta (v_{i} > 0)}{\sum_{i} \delta (v_{i} > 0)}$$

Here $d_{i}$ also describes the euclidean distance between the ground truth keypoint and the detection and $k_{i}$ is a per-keypoint constant that controls falloff.

## Datasets/Benchmarks

What are the most popular datasets in this domain

- Frames Labeled in Cinema (FLIC) (https://bensapp.github.io/flic-dataset.html)
- MPII Human Pose Models (http://pose.mpi-inf.mpg.de/)
- COCO Keypoints/DensePose (https://cocodataset.org/#home)
- Leeds Sports Pose Dataset (https://sam.johnson.io/research/lsp.html) 

## Important Architectures

In the following I will describe a few popular architectures in Single- and Multi-Human Pose Estimation but there are much more out there. Here you can find a great overview of the history of Human Pose Estimation.

### Learning Human Pose Estimation Features with Convolutional Networks (2014)

This paper describes the first deep learning approach to tackle the problem of single-person full body human pose estimation with convolutional neural networks. In this approach the authors trained multiple independent binary classification networks, one network per keypoint. The model is applied in a sliding window approach and outputs a *response map* indicating the confidence of the body part at that location. The figure below shows the architecture of the classification models.

![model architecture of Jain et al. 2014](../assets/imgs/model_architecture_jain_et_al_2014.png)

The input is of shape 64x64 pixel and locally contrast normalized (LCN). As activation functions ReLU is used. To reduce computational complexity max pooling is applied twice which leeds to some spatial information loss. After the three convoliutional layers follow three fully connected layers. To reduce overfitting, L2 regularization and dropout are applied in the fully connected layers. The output layer is a single logistic unit, representing the probability of a body part being in the patch. Moreover the authors use part priors for the final prediction. For a detailed breakdown of how all this works you should have a look at the paper.


### Convolutional Pose Machines (2016)

A Convolutional Pose Machine is a single-person Human Pose Estimation model which incorporates convolutional networks into the pose machine framework from Ramakrishna et al. () and inherits its benefits like the implicit learning of long-range spatial dependencies and a modular sequential design. This results in a differentiable architecture that allows for end-to-end training with backpropagation on large amounts of data. The figure down below shows the overall architecture of the model.

![model architecture of Convolutional Pose Machines](../assets/imgs/conv_pose_machines_architecture.png)

It consists of a sequence of stages (ConvNets) which produce 2D belief maps (heatmaps) for each part/keypoint. Before the images are fed into the network they are scaled down to a size of 368x368 pixels. The first stage consists of seven convolutional and three pooling layers with different kernel sizes. The second and all following stages are different from the first one. Here you use the first layers (share weights) of stage one to produce a belief map which is then concatenatet to the output map of the previous stage. After that you feed the concatenated maps into five more convolutional layers. Every stage outputs P+1 belief maps of 46x46 pixels where P is the number of parts and the additional belief map is for the background. At every stage of the model a loss (MSE) is computed based on these belief maps and divided by the number of pixel values (46x46x15). In the end, these individual losses are added together to form an overall loss. At every stage the prediction quality is refined as you can see in the figure down below.

![joint detections produced by Convolutional Pose Machine model on different stages](../assets/imgs/conv_pose_machines_joint_detections_on_stages.png)

In the first and second stage the model isn't sure which of the two wrists is the right one but in the third stage it seems to be certain. The same goes for the elbows.


### DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model (2016)

depp

### Stacked Hourglass Networks for Human Pose Estimation (2016)

hihi



### Multi-context Attention for Human Pose Estimation (2017)

attetntion



### Learning Feature Pyramids for Human Pose Estimation (2017)

feature pyramids


### RMPE: Regional Multi-person Pose Estimation (2017)

realtime


### Deep High-Resolution Representation Learning for Human Pose Estimation (2019)

jallo

## References

- general information: https://nanonets.com/blog/human-pose-estimation-2d-guide/?utm_source=reddit&utm_medium=social&utm_campaign=pose&utm_content=GROUP_NAME

- cool github with pose estimation papers: https://github.com/cbsudux/awesome-human-pose-estimation

- [1] Convolutional Pose Machines (https://arxiv.org/pdf/1602.00134.pdf)

- [2] 2D Human Pose Estimation: New Benchmark and State of the Art Analysis (http://human-pose.mpi-inf.mpg.de/contents/andriluka14cvpr.pdf)

- [3] Learning Human Pose Estimation Features with Convolutional Networks (https://arxiv.org/pdf/1312.7302.pdf)