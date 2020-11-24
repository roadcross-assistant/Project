# **Road Crossing Assistant**

![main-img](images/roadcross-img.jpg)

<!-- ## 🔮 Table of contents

[1. Introduction](#introduction)

[2. Technologies](#technologies)

[3. Implementation](#implementation)

[4. Project Status](#status) -->

&nbsp;

## 🕶 Introduction

This is a  project with an aim to aid blind people for crossing road, more precisely, Indian roads where there may not be many traffic signals, and where traffic rules and speed limits may not be followed strictly.

We propose a ML/DL based model to **predict safe duration** for crossing road in a given video.

This repository includes Python implementation for various approaches we develop for the prediction model.

<!-- ![roadcross-image](blind-crossing.svg) -->
<!-- &nbsp; -->
## 🕶 Dataset

Our dataset contains 76 videos from diverse locations, time, traffic patterns, collected using an action camera from different roads of Anand, Gujarat.

[Dataset Usage and Download Link](https://docs.google.com/document/d/1uwIMWzDnLLMtVm9TRDQFjIF5yI1wMAf-Fw3d0x39Yvo/edit?usp=sharing)
&nbsp;

## 💻 Technologies

- > **Languages** - Python 
- > **Tools** - VSCode, Anaconda, Jupyter Notebook
- > **Libraries** - Pandas, NumPy, ImageAI, OpenCV


## 🔨 Implementation

### [**Approach 1.1**](https://github.com/roadcross-assistant/Project/blob/master/Approach_1.1_individual_frames.ipynb)

It is our simplest approach where we extracted features from individual frames of the video and trained classification model to predict if a particular frame is safe/unsafe for crossing road.

### [**Approach 1.2**](https://github.com/roadcross-assistant/Project/blob/master/Approach_1.2_individual_frames.ipynb)

It is an advancement over Approach 1.1 where we have made an attempt to improve our feature extraction logic.

*To know more about feature extraction, model training and implementation details, visit our [website](https://rodcross-assistant.github.io/Website/). *


## ⚡ Developed and Contributed by

[Siddhi Brahmbhatt](https://www.github.com/1siddhi7) &nbsp; 🤝 &nbsp; [Yagnesh Patil](https://www.github.com/yagnesh45)
