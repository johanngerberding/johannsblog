---
layout: post
title: Object Detection - Faster Models
date: 2021-05-02 10:45:16 +0200
permalink: /:title
author: Johann Gerberding
tags: object-detection deep-learning
---

***

1. The generated Toc will be an ordered list
{:toc}

***

<br>

## Introduction

<p align="justify">In the previous [post](https://johanngerberding.github.io/johannsblog/Object-Detection-From-R-CNN-to-Mask-RCNN) we have reviewed region-based object detection algorithms (R-CNN models). In the following post I will dive a bit deeper into fast one-stage detection models like YOLO and RetinaNet which are more suited for certain applications with real-time requirements. The models I'm going to talk about here are a bit outdated and don't necessarily correspond to the state-of-the-art in this area anymore. Nevertheless, I find the general development in this area very interesting and the algorithms presented here form the basis for the current state-of-the-art. At that time, two-stage detectors were usually ahead of single-stage detectors in terms of accuracy, which is no longer the case today. In my next post I will go into more detail about state-of-the-art models such as [EfficientDet](https://arxiv.org/pdf/1911.09070.pdf) and [YOLOv4](https://arxiv.org/pdf/2004.10934.pdf).</p>

<br>

## YOLO

<p align="justify">As mentioned before, two stage detection models like Faster R-CNN are region based and considered to slow for certain applications that require real-time capabilities, e.g. in the robotics area or autonomous driving. So let's start with YOLO ("You Only Look Once) which was one of the first approaches to building a fast real-time object detector. Instead of relying on region proposals the authors reframed the object detection as a single regression problem, predicting bounding boxes and class probabilities directly from the images (therefore the name). </p>

![YOLO model architecture]({{ '/assets/imgs/object_detection_2/yolo-network-architecture.png' | relative_url}})**Figure 1.** YOLO network architecture ([source](https://lilianweng.github.io/lil-log/2018/12/27/object-detection-part-4.html "YOLO architecture"))

<p align="justify">This makes the whole system (Figure 1) fairly simple (single ConvNet) and very fast (45 fps). Since the model uses features from the entire image to predict the boxes it reasons globally.</p>

### How it works

<p align="justify">The input image gets divided into an $S \times S$ grid, where each grid cell predicts $B$ bounding boxes and confidence scores ($S=7$,  $B=2$). If the center of an object falls into a grid cell than this grid cell is "responsible" for the detection. Each bounding box consists of 5 predictions: *x_center*, *y_center*, *width*, *height* and *confidence*. The x and y coordinates are relative to the bounds of a grid cell. The width and height are relative to the image. So all predicted values are between 0 and 1. In addition each grid cell also predicts $C$ class probabilities which are conditional on the grid cell containing an object (for PascalVOC: $C=20$). These values encode the probabilities of that class appearing in the box and how well the predicted box fits the object. One of the main limitations of this approach is the fact that each grid cell can only contain one object (max: 49 objects per image).</p>

### Training

<p align="justify">Now let's talk about training the YOLO model. First the convolutional layers get pretrained on ImageNet for a week with an image input size of 224x224. Thereafter to finetune the network on the detection task, four convolutional layers and two fully connected layers get added and the image size is increased to 448x448. As activation LeakyReLU is applied. The loss function is Sum-Squared Error (SSE) consisting of two parts: localization and classification loss:</p>

$$
L_{loc} = \lambda_{coord} \sum_{i=0}^{S^{2}} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} [(x_{i} - \hat{x}_{i})^{2} + (y_{i} - \hat{y}_{i})^{2} + (\sqrt{w_{i}} - \sqrt{\hat{w}_{i}})^{2} + (\sqrt{h_{i}} - \sqrt{\hat{h}_{i}})^{2}]
$$

$$
L_{cls} = \sum_{i=0}^{S^{2}} \sum_{j=0}^{B} (\mathbb{1}_{ij}^{obj} \lambda_{noobj}(1 - \mathbb{1}_{ij}^{obj})) (C_{i} - \hat{C}_{i})^{2} + \sum_{i=0}^{S^{2}} \sum_{C=C} \mathbb{1}_{i}^{obj} (p_{i}(c) - \hat{p}_{i}(c))^{2}
$$

$$
L = L_{cls} + L_{loc}
$$

Symbol | Description
--- | --- 
$\mathbb{1}_{i}^{obj}$ | indicator of whether the cell $i$ contains an object
$\mathbb{1}_{ij}^{obj}$ | ground-truth class label, $u = 1, ..., K$ (background $u = 0$)
$C_{i}$ | confidence score of cell $i$, $Pr(contains object) * IoU (pred, truth)$
$\hat{C}_{i}$ | predicted confidence score (box with higher IoU of the two predicted boxes)
$\mathcal{C}$ | set of all classes (Pascal VOC: 20) 
$p_{i}$ | conditional probability of whether cell $i$ contains an object of class $c \in \mathcal{C}$
$\hat{p}_{i}$ | predicted conditional class probabilities
$S^{2}$ | grid size, here $S=7$
$B$ | number of predicted bounding boxes per grid cell

<p align="justify">Because of model instability due to the inbalance between cells containing or not containing objects the authors use two scale parameters to increase the loss from bounding box predictions ($\lambda_{coord} = 0.5$) and decrease the loss from confidence predictions for boxes that don't contain objects ($\lambda_{noobj} = 0.5$). The loss function only penalizes classification error if an object is present in that grid cell and it only penalizes bbox error if the cell is "responsible" for the ground truth box.</p>

<p align="justify">Some more training details: The authors trained YOLO on VOC 2007 and VOC 2012 with a batch size of 64, momentum of 0.9 and a weight decay of 0.0005. For regularization they rely on dropout and data augmentation.</p>

### Shortcomings

* strong spatial constraints since we have only one prediction per grid cell (7x7 -> max. 49 object predictions); this is one of the reasons why the model struggles with crowds of small objects
* struggles to generalize to objects in new or unusual aspect ratios or configurations (maybe this could be reduced with clever data augmentation or training on different image scales)
* many incorrect localizations due to an inappropriate loss function and coarse features for bounding box prediction (multiple downsampling layers)

<br>

## YOLOv2

<p align="justify">YOLOv2 is basically an improved version of YOLO, adding some tricks to overcome its shortcomings described before. Moreover the paper covers YOLO9000 which is built on top of YOLOv2 and trained with a joint dataset combining COCO and the top 9000 classes of ImageNet (combination of detection and classification). I will only cover YOLOv2 here, for those of you who are interested in YOLO9000 and the joint training procedure, should take a look a the paper.</p>


<p align="justify">As mentioned before, the central goal of YOLOv2 was to fix the problems of YOLO, primarily recall and localization shortcomings. The authors did this based on a variety of new ideas in the field (at that time) and they try to avoid increasing the model size at the same time to preserve the high speed: </p>

<p align="justify">**Batch Normalization:** This leads to significant improvements in convergence while eliminating the need for other forms of regularization like dropout (+2% mAP).</p>

<p align="justify">**High Resolution Classifier:** Finetune the classification network at higher resolution (448x448) for 10 epochs on ImageNet before detection finetuning.</p>

<p align="justify">**Convolutional Anchor Box Detection:** The fully connected layers from YOLO are removed and instead YOLOv2 incorporates anchor boxes (like Faster R-CNN) to predict the bounding boxes; this also decouples the class prediction from the spatial location by predicting class and objectness for every anchor box which leads to a slight decrease in accuracy (-0.3% mAP) but increases recall significantly (+7%) which gives the model more room to improve.</p>

<p align="justify">**Box Dimension Clustering:** Instead of using hand picked anchor box sizes, YOLOv2 runs k-means clustering on the training data to determine good priors for anchor box dimensions; to maximize IoU scores, it relies on the following distance metric:</p>

$$
d(x, c_{i}) = 1 - IoU(x, c_{i}), \quad i=i,...k
$$

where $x$ is a ground truth box candidate and $c_{i}$ is one of the centroids / the closest centroid.


<p align="justify">**Direct Location Prediction:** In Region Proposal Networks the box location prediction is unconstrained which means any anchor box can end up at any point in the image which can lead to an unstable training. YOLOv2 follows the approach of the original YOLO model by predicting location coordinates relative to the location of the grid cell (using a logistic activation). Given the anchor box width $p_{w}$ and height $p_{h}$ in the grid cell with the top left corner ($c_{x}, c_{y}$) the model predicts 5 values ($t_{x}, t_{y}, t_{w}, t_{h}, t_{o}$) which correspond to the following box values:</p>

$$
b_{x} = \sigma (t_{x}) + c_{x} \\
b_{y} = \sigma (t_{y}) + c_{y} \\
b_{w} = p_{w}e^{t_{w}} \\
b_{h} = p_{h}e^{t_{h}} \\
Pr(obj) =IoU(b, obj) = \sigma (t_{o})
$$

This in combination with clustering priors improves mAP by up to 5%.

![YOLOv2 bounding box prediction format]({{ '/assets/imgs/object_detection_2/yolov2-loc.png' | relative_url}}){: style="width: 60%; margin: 0 auto; display: block;"}**Figure 2.** YOLOv2 bounding box prediction ([Redmon, Farhadi](https://arxiv.org/pdf/1612.08242.pdf))

<p align="justify">**Fine-grained Features:** The grid size of the final feature map of YOLOv2 is increased from 7x7 in YOLO to 13x13. Moreover YOLOv2 incorporates a so called passthrough layer that brings features from an earlier layer at 26x26 resolution to the output layer. This process can be compared with identity mappings from ResNets to incorporate higher dimensional features (+1% mAP).</p>

<p align="justify">**Multi-scale Training:** To increase the robustness of the model the authors trained it on images of different sizes. Every 10 batches the input size gets randomly sampled (between 320x320 and 608x608).</p>


<p align="justify">To maintain the high inference speed, YOLOv2 is based on the **Darknet-19** model, consisting of 19 convolutional and 5 max-pooling layers. For detailed information on the architecture check out Table 6 in the [paper](https://arxiv.org/pdf/1612.08242.pdf).</p>

<br>

## RetinaNet

<p align="justify">Next up in our list of fast detection models is RetinaNet. The creators had the goal of closing the accuracy gap between one and two-stage detection approaches. To achieve this, RetinaNet relies on two crucial building blocks, **Feature Pyramid Networks** (FPN) as a backbone and a new loss function called **Focal Loss**.</p>

### Focal Loss

<p align="justify">The central cause for the accuracy gap between the two approaches lies in the extreme foreground-background class imbalance during training. In two-stage detectors this problem is addressed by narrowing down the number of candidate object locations (filtering out many background samples) and by using sampling heuristics like a fixed foreground-to-background ratio or online hard example mining. The proposed Focal Loss is designed to address this issue for one-stage detectors by focusing on hard negatives and down-weighting the easier predictions (obvious empty background). It is based on the normal cross entropy loss (for simplicity we use binary loss down here) </p>

$$CE(p,y) = -y \log p - (1-y) \log (1 - p) $$

<p align="justify">where $y={0,1}$ is a ground truth binary label, indicating whether a bounding box contains an object and $p \in [0,1]$ is the predicted probability that there is an object (also called objectness or confidence score). For notational convenience, let </p>

$$
p_{t} = 
\begin{cases}
    p       & \quad \text{if } y=1\\
    1-p     & \quad \text{otherwise}
\end{cases}
$$

which leads to 

$$
CE(p_{t}) = - \log p_{t}
$$

<p align="justify">Easily classified negatives ($p_{t} \gg 0.5 ,y=0$) comprise the majority of the loss. You can balance the importance of the positive/negative examples by adding a balancing factor $\alpha$</p>

$$
CE(p_{t}) = - \alpha_{t} \log p_{t}
$$

<p align="justify">but this does not differentiate between easy or hard examples. To overcome this the Focal Loss adds a modulating factor $(1-p_{t})^{\gamma}$ with a tunable focusing parameter $\gamma \geq 0$:</p>

$$
FL(p_{t}) = - (1 - p_{t})^{\gamma} \log (p_{t})
$$

![Focal Loss with different gamma values]({{ '/assets/imgs/object_detection_2/focal-loss.png' | relative_url}}){: style="width: 70%; margin: 0 auto; display: block;"}**Figure 3.** Focal Loss with different gamma values ([Lin et al.](https://arxiv.org/pdf/1708.02002.pdf "Focal Loss with different gamma values"))

<p align="justify">For better control of the shape of the weighting function the authors used an $\alpha$-balanced version in practice, where $\alpha = 0.25$ and $\gamma = 2$ worked best in their experiments:</p>

$$
FL(p_{t}) = - \alpha (1 - p_{t})^{\gamma} \log (p_{t})
$$


### Feature Pyramid Network

<p align="justify">
The FPN backbone for RetinaNet was constructed on top of ResNet. To really understand what that means you should take a look at the [paper](https://arxiv.org/pdf/1612.03144.pdf). Figure 4 down below shows the fundamental idea of FPN which is to leverage a ConvNets pyramidal feature hierarchy to build a feature pyramid with high level semantics throughout. It is general purpose and can be applied to many convolutional backbone architectures.
</p>

![Featurized Pyramid Network architecture]({{ '/assets/imgs/object_detection_2/featurized-image-pyramid.png' | relative_url}}){: style="width: 100%; margin: 0 auto; display: block;"}**Figure 4.** Featurized Pyramid Network architecture ([source](https://lilianweng.github.io/lil-log/2018/12/27/object-detection-part-4.html#yolo-you-only-look-once "Featurized Pyramid Network architecture"))

<p align="justify">
The basic structure contains a sequence of pyramid levels each corresponding to one network stage. Often these stages contain multiple conv layers of the same size and stage sizes are scaled down by a factor of two. $C_{i}$ represents the different layers of those stages (for ResNet e.g. {$C_{2}, C_{3}, C_{4}, C_{5}$}). As you can see there are two different pathways which connect the conv layers:</p>

1. **Bottom-up** pathway: regular feedback path
2. **Top-down** pathway: goes in the opposite direction, adding coarse but semantically stronger feature maps back into the previous levels of layer size by lateral connections (1x1 conv to match dimensions) and nearest neighbor upsampling; the combination of the two maps is done by element-wise addition

<p align="justify">The final predictions ({$P_{i}$} where $i$ indicates the pyramid level and has resolution $2^{i}$ lower than the input) are_ generated out of every merged map by a 3x3 conv layer. RetinaNet utilizes feature pyramid levels $P_{3}$ to $P_{7}$ computed from the corresponding ResNet residual stage from $C_{3}$ to $C_{5}$. All pyramid levels have 256 channels (most of RetinaNet is similar to FPN with a few minor differences). The authors used translation-invariant anchor boxes as priors, similar to those used in RPN variant of FPN. To improve Average Precision the number of anchors was increased to $A=9$ (three aspect ratios {1:2, 1:1, 2:1} with three different sizes {$2^{0}, 2^{1/3}, 2^{2/3}$}). As seen before, for each anchor box the model predicts a class probability for each of $K$ classes with a classification subnet trained with Focal Loss. A box regression subnet outputs the offsets for the boxes to the nearest ground truth object. Both networks are independent Fully Convolutional Networks that don't share any parameters.</p>

![RetinaNet architecture]({{ '/assets/imgs/object_detection_2/retina-net.png' | relative_url}}){: style="width: 100%; margin: 0 auto; display: block;"}**Figure 5.** RetinaNet architecture ([source](https://lilianweng.github.io/lil-log/2018/12/27/object-detection-part-4.html#yolo-you-only-look-once "RetinaNet architecture"))

<br>

## YOLOv3

<p align="justify">
YOLOv3 was created by applying changes to YOLOv2 inspired by, at that time, recent advances in the object detection world. It's a pretty short and rather unscientifically (I like it :D ) written. The following list summarizes the most important improvements:
</p>

* **Logistic Regression for objectness scores** instead of sum of squared errors
* **Independent Logistic Classifiers** for class prediction instead of softmax which increases the performance on non mutually exclusive multilabel datasets like Open Images
* **Multi-scale predictions** inspired by FPN (3 scales per stage)
* **Darknet-53 as Feature Extractor** which performs similar to ResNet-152 but is 2x faster

<p align="justify">
Overall YOLOv3 performs better and faster than SSD, worse then RetinaNet but is 3.8x faster and comparable to state-of-the-art methods on the $AP_{50}$ metric at that time. In the appendix Joseph (the author) adds a cool comment on his opinion about the COCO evaluation metrics. It's refreshing to see someone questioning stuff like this.</p>

![YOLOv3 performance]({{ '/assets/imgs/object_detection_2/yolov3-res.png' | relative_url}}){: style="width: 80%; margin: 0 auto; display: block;"}**Figure 6.** YOLOv3 performance ([Redmon, Farhadi](https://arxiv.org/pdf/1804.02767.pdf "YOLOv3 performance"))

<br>

## Summary

<p align="justify">In this blog post, we went over four popular but now somewhat aging fast object recognition systems and you got a first introduction to the world of real-time object recognition. In the next post I would like to talk about some more recent models like EfficientDet and YOLOv4.</p>

## References

[[1]](https://arxiv.org/pdf/1506.02640.pdf) Joseph Redmon, et al. “You only look once: Unified, real-time object detection.” CVPR 2016.

[[2]](https://arxiv.org/pdf/1612.08242.pdf) Joseph Redmon and Ali Farhadi. “YOLO9000: Better, Faster, Stronger.” CVPR 2017.

[[3]](https://arxiv.org/pdf/1804.02767.pdf) Joseph Redmon, Ali Farhadi. “YOLOv3: An incremental improvement.”.

[[4]](https://arxiv.org/pdf/1612.03144.pdf) Tsung-Yi Lin, et al. “Feature Pyramid Networks for Object Detection.” CVPR 2017.

[[5]](https://arxiv.org/pdf/1708.02002.pdf) Tsung-Yi Lin, et al. “Focal Loss for Dense Object Detection.” IEEE transactions on pattern analysis and machine intelligence, 2018.