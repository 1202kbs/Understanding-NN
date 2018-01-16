# Understanding NN

Tensorflow walkthrough of [Methods for Interpreting and Understanding Deep Neural Networks](https://arxiv.org/abs/1706.07979).


## 1 Interpreting a DNN Model

*This section corresponds to Section 3 in the original paper.*

This section focuses on the problem of interpreting a concept learned by a deep neural network (DNN).


### 1.1 Activation Maximization (AM)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/1_1_Activation_Maximization/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/1_1_Activation_Maximization/DNN_2.png)


### 1.3 Performing AM in Code Space

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/1_3_AM_Code/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/1_3_AM_Code/DNN_2.png)


## 2 Explaining DNN Decisions

*This section corresponds to Section 4 in the original paper.*

In this section, we ask for a given data point x, what makes it representative of a certain concept encoded in some output neuron of the deep neural network (DNN).


### 2.1 Sensitivity Analysis

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_1_SA/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_1_SA/DNN_2.png)


### 2.2 Simple Taylor Decomposition

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_2_STD/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_2_STD/DNN_2.png)


## 3 The LRP Explanation Framework

*This section corresponds to Section 5 in the original paper.*

In this section, we focus on the [layer-wise relevance propagation](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140) (LRP) technique introduced by Bach et al. and the [Deep Taylor Decomposition](https://www.sciencedirect.com/science/article/pii/S0031320316303582) technique introduced by Montavon et al. for explaining
deep neural network decisions.


### 3.1 Layer-wise Relevance Propagation (Work Under Progress)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/3_1_LRP/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/3_1_LRP/DNN_2.png)


### 3.2 Deep Taylor Decomposition (Work Under Progress)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/3_2_DTD/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/3_2_DTD/DNN_2.png)


## 4 Quantifying Explanation Quality

*This section corresponds to Section 7 in the original paper.*

In Sections 2 and 3 (Sections 4 and 5 in the original paper), we have introduced a number of explanation techniques. While each technique is based on its own intuition or mathematical principle, it is also important to dene at a more abstract level what are the characteristics of a good explanation, and to be able to test for these characteristics quantitatively. A quantitative framework allows to compare explanation techniques specifically for a target
problem, e.g. ILSVRC or MIT Places. We present in Sections 4.1 and 4.2 two important properties of an explanation, along with possible evaluation metrics.


### 4.1 Explanation Continuity

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/4_1_EC/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/4_1_EC/graph.png)


### 4.2 Explanation Selectivity

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/4_2_ES/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/4_2_ES/DNN_2.png)

<p align="center">
  <img src="https://github.com/1202kbs/Understanding-NN/blob/master/assets/4_2_ES/graph.png" alt="Explanation Technique Comparison Graph"/>
</p>


## Prerequisites

This code requires [Tensorflow](https://www.tensorflow.org/), [NumPy](http://www.numpy.org/) and [OpenCV](https://opencv.org/).


## Usage

Step-by-step tutorials are implemented in Jupyter Notebooks.
