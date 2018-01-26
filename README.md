# Understanding NN

This repository is intended to be a self-contained tutorial of various DNN interpretation and explanation techniques. Explanation of the theoretical background as well as step-by-step Tensorflow implementation for practical usage are both covered in the Jupyter Notebooks.

**UPDATE**

It seems that Github is unable to render some of the equations in the notebooks. I strongly recommend using the nbviewer until I find out what the problem is (you can also download the repo and view them on your local environment). Links are listed below.


## Nbviewer Links

[1.1 Activation Maximization](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/1.1%20Activation%20Maximization.ipynb)

[1.3 Performing AM in Code Space](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/1.3%20Performing%20AM%20in%20Code%20Space.ipynb)

[2.1 Sensitivity Analysis](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/2.1%20Sensitivity%20Analysis.ipynb)

[2.2 Simple Taylor Decomposition](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/2.2%20Simple%20Taylor%20Decomposition.ipynb)

[2.3 Deconvolution](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/2.3%20Deconvolution.ipynb)

[2.4 Backpropagation](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/2.4%20Backpropagation.ipynb)

[3.1 Layer-wise Relevance Propagation Part 1](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/3.1%20Layer-wise%20Relevance%20Propagation%20%281%29.ipynb)

[3.1 Layer-wise Relevance Propagation Part 2](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/3.1%20Layer-wise%20Relevance%20Propagation%20%282%29.ipynb)

[3.2 Deep Taylor Decomposition Part 1](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/3.2%20Deep%20Taylor%20Decomposition%20%281%29.ipynb)

[3.2 Deep Taylor Decomposition Part 2](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/3.2%20Deep%20Taylor%20Decomposition%20%282%29.ipynb)

[4.1 Explanation Continuity](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/4.1%20Explanation%20Continuity.ipynb)

[4.2 Explanation Selectivity](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/4.2%20Explanation%20Selectivity.ipynb)


## 1 Interpreting a DNN Model

This section focuses on the problem of interpreting a concept learned by a deep neural network (DNN).


### 1.1 Activation Maximization (AM)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/1_1_Activation_Maximization/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/1_1_Activation_Maximization/DNN_2.png)


### 1.3 Performing AM in Code Space

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/1_3_AM_Code/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/1_3_AM_Code/DNN_2.png)


## 2 Explaining DNN Decisions

In this section, we ask for a given data point x, what makes it representative of a certain concept encoded in some output neuron of the deep neural network (DNN).


### 2.1 Sensitivity Analysis

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_1_SA/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_1_SA/DNN_2.png)


### 2.2 Simple Taylor Decomposition

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_2_STD/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_2_STD/DNN_2.png)


### 2.3 Deconvolution

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_3_DC/layer1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_3_DC/layer2.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_3_DC/layer3.png)


### 2.4 Backpropagation

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_4_BP/prototype1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_4_BP/prototype2.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_4_BP/saliency1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_4_BP/saliency2.png)


## 3 The LRP Explanation Framework

In this section, we focus on the [layer-wise relevance propagation](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140) (LRP) technique introduced by Bach et al. and the [Deep Taylor Decomposition](https://www.sciencedirect.com/science/article/pii/S0031320316303582) technique introduced by Montavon et al. for explaining
deep neural network decisions.


### 3.1 Layer-wise Relevance Propagation

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/3_1_LRP/fig.png)


### 3.2 Deep Taylor Decomposition

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/3_2_DTD/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/3_2_DTD/DNN_2.png)


## 4 Quantifying Explanation Quality

In Sections 2 and 3 (Sections 4 and 5 in the original paper), we have introduced a number of explanation techniques. While each technique is based on its own intuition or mathematical principle, it is also important to define at a more abstract level what are the characteristics of a good explanation, and to be able to test for these characteristics quantitatively. We present in Sections 4.1 and 4.2 two important properties of an explanation, along with possible evaluation metrics.


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


## References

#### Sections 1.1 ~ 2.2 and 4.1 ~ 4.2

[1] Montavon, G., Samek, W., Müller, K., jun 2017. Methods for Interpreting and Understanding Deep Neural Networks. arxiv preprint, arXiv:1706.07979.

#### Section 1.3

[2] Nguyen, A., Dosovitskiy, A., Yosinski, J., Brox, T., Clune, J., 2016. Synthesizing the preferred inputs for neurons in neural networks via deep generator networks. In: Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain. pp. 3387-3395.

[3] A. Dosovitskiy and T. Brox. Generating images with perceptual similarity metrics based on deep networks. In NIPS, 2016.

#### Section 2.3

[4] Zeiler, M. D., Fergus, R., 2014. Visualizing and understanding convolutional networks. In: Computer Vision - ECCV 2014 - 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part I. pp. 818-833.

#### Section 2.4

[5]  K. Simonyan, A. Vedaldi, and A. Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. In Workshop at International Conference on Learning Representations, 2014.

#### Section 3.1

[6] Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.R., Samek, W., 07 2015. On pixel-wise explanations for non-linear classier decisions by layer-wise relevance propagation. PLOS ONE 10 (7), 1-46.

#### Section 3.2

[7] Montavon, G., Lapuschkin, S., Binder, A., Samek, W., Müller, K.R., 2017. Explaining nonlinear classication decisions with deep Taylor decomposition. Pattern Recognition 65, 211-222.
