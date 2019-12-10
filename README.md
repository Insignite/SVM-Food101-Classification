# SVM Food101 Dataset Classification

## Introduction
**Support Vector Machines (SVM)** is a supervised learning model with associated algorithms that analyzes data by plotting data points on N-dimensionals graph (N is the number of features) and performs classification by drawing an optimal hyperplane. Data points that closer to the hyperplane influence the position and the orientation of the hyperplane. With this information, we can optimize the hyperplane by fine tuning **Cost (C)** and **Gradient (g = gamma substitute variable)**. Large **C** decreases the margin of the hyperplane, which allow much less misclassified points and lead to hyperplane attemp to fit as many point as possible, where as small **C** allows more generalization and smoother hyperplane. For **g**, a higher value leads to a lower Ecludien distance between data points and scale down fit area.



I built a SVM classification with two approach: 
####Histogram of Oriented Gradients (HOG)
![hog]()
####Transfer learning. 

I want to build a simple Deep Learning model for image classification on Kaggle Dog vs. Cat Dataset. In this project, I decided to use AlexNet architecture as it repeatedly mention during my Machine Learning course. This project is simple enough that helps me understand Alexnet, familiarize with Keras, and gain more experience in ML field.
