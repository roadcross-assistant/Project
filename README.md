# **Road Crossing Assistant**

![main-img](images/roadcross-img.jpg)

## 💡 Introduction

Roads in medium-sized Indian towns often have lots of traffic but no (or disregarded) traffic stops. This makes it hard for the blind to cross roads safely, because vision is crucial to determine when crossing is safe. Automatic and reliable image-based safety classifiers thus have the potential to help the blind to cross Indian roads. Yet, we currently lack datasets collected on Indian roads from the pedestrian point-of-view, labelled with road crossing safety information. Existing classifiers from other countries are often intended for crossroads, and hence rely on the detection and presence of traffic lights, which is not applicable in Indian conditions. We introduce INDRA (INdian Dataset for RoAd crossing), the first dataset capturing videos of Indian roads from the pedestrian point-of-view. INDRA contains 104 videos comprising of 26k 1080p frames, each annotated with a binary road crossing safety label and vehicle bounding boxes. We train various classifiers to predict road crossing safety on this data, ranging from SVMs to convolutional neural networks (CNNs). The best performing model DilatedRoadCrossNet is a novel single-image architecture tailored for deployment on the Nvidia Jetson Nano. It achieves 79% recall at 90% precision on unseen images. Lastly, we present a wearable road crossing assistant running DilatedRoadCrossNet, which can help the blind cross Indian roads in real-time.

This repository includes Python implementation for various approaches we developed for the prediction model.

⭐ Awarded Best Paper (Indian Context) at [ICVGIP 2022](https://events.iitgn.ac.in/2022/icvgip/). Find the arXiv pre-print [here](https://arxiv.org/abs/2211.07916).

⭐ Awarded financial assistance under the Innovation track of the Govt. of Gujarat’s [Startups & Innovation Policy](http://www.ssipgujarat.in/new_student1.php).

&nbsp;

## 🗃 Dataset

Any suitable dataset did not exist (datasets for autonomous cars are not recorded from a pedestrian’s point of view), so we have created our own dataset. Our dataset contains 104 videos from diverse locations, time, traffic patterns, collected using an action camera from different roads of Anand, Gujarat.

[Dataset Usage and Download Link ![Dataset-link](https://img.icons8.com/fluent-systems-filled/15/0366D6/external-link.png)](https://www.kaggle.com/datasets/siddhi17/road-crossing-dataset)

&nbsp;

<!-- ## 💻 Technologies

- > **Languages** - Python
- > **Tools** - VSCode, Anaconda, Jupyter Notebook
- > **Libraries** - Tensorflow, ImageAI, Pandas, NumPy, OpenCV

&nbsp; -->

## 🔨 Implementation

### 1. Machine Learning - Single Frame SVM


#### [**Approach 1.1 (precision : 0.50, recall : 0.70)** ![Approach 1.1](https://img.icons8.com/fluent-systems-filled/18/0366D6/external-link.png)](https://github.com/roadcross-assistant/Project/blob/master/ML/Approach_1.1_individual_frames.ipynb)

It is our simplest approach where we extracted simple per-frame features capturing number, location and size of vehicles. We used SVM to train the classification model. 

#### [**Approach 1.2 (precision : 0.54 , recall : 0.74)** ![Approach 1.2](https://img.icons8.com/fluent-systems-filled/18/0366D6/external-link.png)](https://github.com/roadcross-assistant/Project/blob/master/ML/Approach_1.2_individual_frames.ipynb)

It is an advancement over Approach 1.1 where we have improved our feature extraction logic by ignoring the vehicles traveling on the opposite half of the road (using vehicle tracking)

#### [**Approach 1.3 (precision : 0.68 , recall : 0.87)** ![Approach 1.3](https://img.icons8.com/fluent-systems-filled/18/0366D6/external-link.png)](https://github.com/roadcross-assistant/Project/blob/master/ML/Approach_1.3_individual_frames.ipynb)

It is an advancement over Approach 1.2 where we have improved our feature extraction logic by considereing relative speed of the vehicles, and we also improved labels by annotating the videos frame-wise (instead of second-wise).

&nbsp;

### 2. Machine Learning - Multi Frame SVM

#### [**Approach 2.1 (precision : 0.74 , recall : 0.88)** ![Approach 2.1](https://img.icons8.com/fluent-systems-filled/18/0366D6/external-link.png)](https://github.com/roadcross-assistant/Project/blob/master/ML/Approach_2.1_multiple_frames.ipynb)

As it is obvious that even we as humans do not decide whether it is safe to cross a road by just having one glance at the road, we have started using multi-frame features instead of individual-frame features in this approach.

#### [**Approach 2.2 (precision : 0.79 , recall : 0.83)** ![Approach 2.2](https://img.icons8.com/fluent-systems-filled/18/0366D6/external-link.png)](https://github.com/roadcross-assistant/Project/blob/master/ML/Approach_2.2_multiple_frames.ipynb)

It is similar to Approach 2.1, in which we have used multi-frame features in a sliding window based manner. Its feature extraction logic is a bit optimized as compared to that of Approach 2.1.

&nbsp;

### 3. Deep Learning - Single frame CNN


#### [**Approach 3.1 (precision : 0.90 , recall : 0.60)** ![Approach 3.1](https://img.icons8.com/fluent-systems-filled/18/0366D6/external-link.png)](https://github.com/roadcross-assistant/Project/blob/master/DL/Approach_3.1.py)

In Approach 3.1, we used the MobileNetV2 architecture with additional dense layers at the top. We used the MobileNetV2 because it is a lightweight architecture particularly useful for mobile and embedded vision applications.

#### [**Approach 3.2 (precision : 0.90 , recall : 0.73)** ![Approach 3.2](https://img.icons8.com/fluent-systems-filled/18/0366D6/external-link.png)](https://github.com/roadcross-assistant/Project/blob/master/DL/Approach_3.2.py)

As Approach 3.1 did not give a satisfactory performance on test data, in Approach 3.2 we have developed our own CNN architecture from scratch. However, since it's convolutional layers consisted of kernels of size greater than 3, its inference speed was very low even after optimization to a TensorRT graph.

#### [**Approach 3.3 (precision : 0.90 , recall : 0.77)** ![Approach 3.3](https://img.icons8.com/fluent-systems-filled/18/0366D6/external-link.png)](https://github.com/roadcross-assistant/Project/blob/master/DL/Approach_3.3.py)

It is an advancement over approach 3.2, in which we have replaced the convolutional layers with larger kernal size with dilated convolutional layers. This resulted in a higher inference speed after optimization to TensorRT graph.

&nbsp;

Note : The precision and recall mentioned are model performances on test data.

*To know more about feature extraction, model training and implementation details, visit our [website ![website-link](https://img.icons8.com/fluent-systems-filled/15/0366D6/external-link.png)](https://roadcross-assistant.github.io/Website/ "Road Crossing Assistant Website").*

&nbsp;

## ⚡ Developed and Contributed by

[Siddhi Brahmbhatt](https://www.github.com/1siddhi7) &nbsp; 🤝 &nbsp; [Yagnesh Patil](https://www.github.com/yagnesh45)
