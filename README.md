# **Road Crossing Assistant**

![main-img](images/roadcross-img.jpg)

## 💡 Introduction

This is a  project with an aim to aid blind people for crossing road, more precisely, Indian roads where there may not be many traffic signals, and where traffic rules and speed limits may not be followed strictly.

We propose a ML/DL based model to **predict safe duration** for crossing road in a given video.

This repository includes Python implementation for various approaches we develop for the prediction model.

&nbsp;

## 🗃 Dataset

Our dataset contains 104 videos from diverse locations, time, traffic patterns, collected using an action camera from different roads of Anand, Gujarat.

[Dataset Usage and Download Link ![Dataset-link](https://img.icons8.com/fluent-systems-filled/15/0366D6/external-link.png)](https://forms.gle/xtkZexnEuRVYfwAT6)

&nbsp;

## 💻 Technologies

- > **Languages** - Python
- > **Tools** - VSCode, Anaconda, Jupyter Notebook
- > **Libraries** - Pandas, NumPy, ImageAI, OpenCV

&nbsp;

## 🔨 Implementation

### 1. Machine Learning - Single Frame SVM

In this approach, We are using the classfication model - SVM for predicting safe/unsafe frames of the videos.

#### [**Approach 1.1** ![Approach 1.1](https://img.icons8.com/fluent-systems-filled/18/0366D6/external-link.png)](https://github.com/roadcross-assistant/Project/blob/master/ML/Approach_1.1_individual_frames.ipynb)

It is our simplest approach where we extracted features from individual frames of the video and trained classification model to predict if a particular frame is safe/unsafe for crossing road.

#### [**Approach 1.2** ![Approach 1.2](https://img.icons8.com/fluent-systems-filled/18/0366D6/external-link.png)](https://github.com/roadcross-assistant/Project/blob/master/ML/Approach_1.2_individual_frames.ipynb)

It is an advancement over Approach 1.1 where we have made an attempt to improve our feature extraction logic using direction detection.

#### [**Approach 1.3** ![Approach 1.3](https://img.icons8.com/fluent-systems-filled/18/0366D6/external-link.png)](https://github.com/roadcross-assistant/Project/blob/master/ML/Approach_1.3_individual_frames.ipynb)

It is an advancement over Approach 1.2 where we have made an attempt to improve our features by adding relative speed of vehicles as features, and manually labelling the videos frame-wise instead of second-wise.

&nbsp;

### 2. Machine Learning - Multi Frame SVM

#### [**Approach 2.1** ![Approach 2.1](https://img.icons8.com/fluent-systems-filled/18/0366D6/external-link.png)](https://github.com/roadcross-assistant/Project/blob/master/ML/Approach_2.1_multiple_frames.ipynb)

As it is obvious that even we as humans do not decide whether it is safe to cross a road by just having one glance at the road, we have started using multi-frame features instead of individual-frame features in this approach.

#### [**Approach 2.2** ![Approach 2.2](https://img.icons8.com/fluent-systems-filled/18/0366D6/external-link.png)](https://github.com/roadcross-assistant/Project/blob/master/ML/Approach_2.2_multiple_frames.ipynb)

Approach 2.2 is similar to Approach 2.1, in which we have used multi-frame features in a sliding window based manner. Its feature extraction logic is a bit different as compared to Approach 2.1.

&nbsp;

### 3. Deep Learning - Single frame CNN

Deep Learning has been shown to learn highly effective features from image and video data, yielding high accuracy in many tasks. Therefore in this approach we will be extensively working on different CNN architectures and also perform training experiments.

#### [**Approach 3.1** ![Approach 3.1](https://img.icons8.com/fluent-systems-filled/18/0366D6/external-link.png)](https://github.com/roadcross-assistant/Project/blob/master/DL/Approach_3.1.py)

Approach 3.1 is developed based on the SSD-MobileNetv2 neural network. Also, building the input pipeline for our project required building a custom dataset using tensorflow’s dataset module (tf.data) since tensorflow does not have an option to generate baches of video data.

#### [**Approach 3.2** ![Approach 3.2](https://img.icons8.com/fluent-systems-filled/18/0366D6/external-link.png)](https://github.com/roadcross-assistant/Project/blob/master/DL/Approach_3.2.py)

Approach 3.2 consists of custom-light weight model architecture built and trained from the scratch to make it useful for this project's application-specific purpose rather than using standard general-purpose model.

#### [**Approach 3.3** ![Approach 3.3](https://img.icons8.com/fluent-systems-filled/18/0366D6/external-link.png)](https://github.com/roadcross-assistant/Project/blob/master/DL/Approach_3.3.py)

Approach 3.3 is continuation of Approach 3.2 custom model, that is developed to work as faster, efficient and light-weight model architecture to run on small computing devices with the help of dilated convolutions.

&nbsp;

*To know more about feature extraction, model training and implementation details, visit our [website ![website-link](https://img.icons8.com/fluent-systems-filled/15/0366D6/external-link.png)](https://roadcross-assistant.github.io/Website/ "Road Crossing Assistant Website").*

&nbsp;

## ⚡ Developed and Contributed by

[Siddhi Brahmbhatt](https://www.github.com/1siddhi7) &nbsp; 🤝 &nbsp; [Yagnesh Patil](https://www.github.com/yagnesh45)
