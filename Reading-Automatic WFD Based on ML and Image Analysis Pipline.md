https://link.springer.com/content/pdf/10.1007%2F978-3-030-16184-2_24.pdf

Author: Joao Alves, Christophe Soares, etc.\

# Abstract
Detect forest fire in early stage.\
To detect the presence of smoke or flame.\
To produce an estimation of the area under ignotion so that its size  can be evaluateed.\
The organized dataset, composed by 882 images, was associated with relevant image metadata.\

# 1. Introduction
Overall in crease in temperature by $2^oC$ cause higher level of dryness in the environment.\
Among the available fire detection techniques, we emphasize the use of video cameras as **input source**.\
* Aim: identifing a fire situation.
* Requires: 2 initial steps
  * 1. extraction of image descriptors
  * 2. feeds a posterior classification.
* Outputs: Together, these extraction and classification phases form the pipline of a *Deep Learning* model. The extraction
of discriptors return s unique properties that characterize an image by mean of a numerical vector.This vector is the output of a DCNN module. 
* The obtained descriptors are then used to train a ML classifier.
* After obtaining a fire classification, the pipline focuses on spotting the image areas with flames, through the application of CV.

# 2. State of Art
Projects that propose the use of mobile or static video camera in fire detection.\
(1) based on CV techniques, only seek to detect flames or smoke; (2) using DL techniques, no such restriction.\
## 2.1 DL in Fire Detection (FD)
Best choice of optimizers, reduction functions and learning rate: Inception-V3. *Adam* optimizer, *reduce max* function and *0.1* learning rate. [6]\
Two CNN to improve detection accuracy by fine-tuning the fully connected layers of both. VGG16 and Res-Net50, accuracy improvement **1%**. [7]\
Five DCNNs are studied: Alex-Net, Google-Net, modified Google-Net, VGG13 and modified VGG13. Concluded that: Goole-Net > modified VGG13 > others.
![comparing different DNNs](/comparing.png)
## 2.2 CV System in Fire Detection
**Flame Detection:**\ 
Suitable for daytime scenarios and is **intended to be applied to drones**. Used a new color index called FFDI [9].\
By manipulating RGB and YCrCb color spaces. Algorithms consists of seven rules to be applied sequentially [10].\
To improve the accuracy on fixed surveillance cameras. Two parallel image streams, one for detection and the other for confirmation of the  results. [11]\
**Smoke Detection:**\




