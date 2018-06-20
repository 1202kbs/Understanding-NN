# Understanding NN

This repository is intended to be a tutorial of various DNN interpretation and explanation techniques. Explanation of the theoretical background as well as step-by-step Tensorflow implementation for practical usage are both covered in the Jupyter Notebooks. I did not include explanation for techniques for which I thought the algorithm as well as the explanation of the original paper was clear.

**UPDATE**

It seems that Github is unable to render some of the equations in the notebooks. I strongly recommend using the nbviewer until I find out what the problem is (you can also download the repo and view them on your local environment). Links are listed below.

## Nbviewer Links

[1.1 Activation Maximization](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/1.1%20Activation%20Maximization.ipynb)

[1.3 Performing AM in Code Space](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/1.3%20Performing%20AM%20in%20Code%20Space.ipynb)

[2.1 Sensitivity Analysis](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/2.1%20Sensitivity%20Analysis.ipynb)

[2.2 Simple Taylor Decomposition](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/2.2%20Simple%20Taylor%20Decomposition.ipynb)

[2.3 Layer-wise Relevance Propagation Part 1](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/2.3%20Layer-wise%20Relevance%20Propagation%20%281%29.ipynb)

[2.3 Layer-wise Relevance Propagation Part 2](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/2.3%20Layer-wise%20Relevance%20Propagation%20%282%29.ipynb)

[2.4 Deep Taylor Decomposition Part 1](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/2.4%20Deep%20Taylor%20Decomposition%20%281%29.ipynb)

[2.4 Deep Taylor Decomposition Part 2](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/2.4%20Deep%20Taylor%20Decomposition%20%282%29.ipynb)

[2.5 DeepLIFT](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/2.5%20DeepLIFT.ipynb)

[3.1 Deconvolution](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/3.1%20Deconvolution.ipynb)

[3.2 Backpropagation](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/3.2%20Backpropagation.ipynb)

[3.3 Guided Backpropagation](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/3.3%20Guided%20Backpropagation.ipynb)

[3.4 Integrated Gradients](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/3.4%20Integrated%20Gradients.ipynb)

[3.5 SmoothGrad](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/3.5%20SmoothGrad.ipynb)

[4.1 Class Activation Map](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/4.1%20CAM.ipynb)

[4.2 Grad-CAM](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/4.2%20Grad-CAM.ipynb)

[4.2 Grad-CAM++](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/4.3%20Grad-CAM%2B%2B.ipynb)

[5.1 Explanation Continuity](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/5.1%20Explanation%20Continuity.ipynb)

[5.2 Explanation Selectivity](http://nbviewer.jupyter.org/github/1202kbs/Understanding-NN/blob/master/5.2%20Explanation%20Selectivity.ipynb)


## 1 Activation Maximization

This section focuses on interpreting a concept learned by a deep neural network (DNN) through activation maximization.


### 1.1 Activation Maximization (AM)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/1_1_Activation_Maximization/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/1_1_Activation_Maximization/DNN_2.png)


### 1.3 Performing AM in Code Space

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/1_3_AM_Code/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/1_3_AM_Code/DNN_2.png)


## 2 Layer-wise Relevance Propagation

In this section, we first introduce the concept of relevance score with Sensitivity Analysis, explore basic relevance decomposition with Simple Taylor Decomposition and then build up to various Layer-wise Relevance Propagation methods such as Deep Taylor Decomposition and DeepLIFT.


### 2.1 Sensitivity Analysis

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_1_SA/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_1_SA/DNN_2.png)


### 2.2 Simple Taylor Decomposition

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_2_STD/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_2_STD/DNN_2.png)


### 2.3 Layer-wise Relevance Propagation

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_3_LRP/fig.png)


### 2.4 Deep Taylor Decomposition

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_4_DTD/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_4_DTD/DNN_2.png)


### 2.5 DeepLIFT

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_5_DL/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/2_5_DL/DNN_2.png)


## 3 Gradient Based Methods

Implementation of various types of gradient-based visualization methods such as Deconvolution, Backpropagation, Guided Backpropagation, Integrated Gradients and SmoothGrad. Check out [grad.py](https://github.com/1202kbs/Understanding-NN/blob/master/models/grad.py), a modular implementation of various gradient-based visualization techniques.


### 3.1 Deconvolution

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/3_1_DC/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/3_1_DC/DNN_2.png)


### 3.2 Backpropagation

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/3_2_BP/saliency1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/3_2_BP/saliency2.png)


### 3.3 Guided Backpropagation

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/3_3_GBP/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/3_3_GBP/DNN_2.png)


### 3.4 Integrated Gradients

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/3_4_IG/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/3_4_IG/DNN_2.png)


### 3.5 SmoothGrad

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/3_5_SG/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/3_5_SG/DNN_2.png)


## 4 Class Activation Map

Implementation of Class Activation Map (CAM) and its generalized versions, Grad-CAM and Grad-CAM++ the [cluttered MNIST](https://github.com/deepmind/mnist-cluttered) dataset.


### 4.1 Class Activation Map

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/4_1_CAM/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/4_1_CAM/DNN_2.png)


### 4.2 Grad-CAM

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/4_2_GCAM/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/4_2_GCAM/DNN_2.png)


### 4.3 Grad-CAM++

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/4_3_GCAMPP/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/4_3_GCAMPP/DNN_2.png)


## 5 Quantifying Explanation Quality

While each explanation technique is based on its own intuition or mathematical principle, it is also important to define at a more abstract level what are the characteristics of a good explanation, and to be able to test for these characteristics quantitatively. We present in Sections 5.1 and 5.2 two important properties of an explanation, along with possible evaluation metrics.


### 5.1 Explanation Continuity

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/5_1_EC/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/5_1_EC/graph.png)


### 5.2 Explanation Selectivity

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/5_2_ES/DNN_1.png)

![alt tag](https://github.com/1202kbs/Understanding-NN/blob/master/assets/5_2_ES/DNN_2.png)

<p align="center">
  <img src="https://github.com/1202kbs/Understanding-NN/blob/master/assets/5_2_ES/graph.png" alt="Explanation Technique Comparison Graph"/>
</p>


## Prerequisites

This tutorial requires [Tensorflow](https://www.tensorflow.org/), [NumPy](http://www.numpy.org/), [Matplotlib](https://matplotlib.org/), and [OpenCV](https://opencv.org/).


## References

#### Sections 1.1 ~ 2.2 and 5.1 ~ 5.2

[1] Montavon, G., Samek, W., Müller, K., jun 2017. Methods for Interpreting and Understanding Deep Neural Networks. arXiv preprint arXiv:1706.07979, 2017.

#### Section 1.3

[2] Nguyen, A., Dosovitskiy, A., Yosinski, J., Brox, T., Clune, J., 2016. Synthesizing the preferred inputs for neurons in neural networks via deep generator networks. In: Advances in Neural Information Processing Systems 29: Annual Conference on Neural Information Processing Systems 2016, December 5-10, 2016, Barcelona, Spain. pp. 3387-3395.

[3] A. Dosovitskiy and T. Brox. Generating images with perceptual similarity metrics based on deep networks. In NIPS, 2016.

#### Section 2.3

[4] Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.R., Samek, W., 07 2015. On pixel-wise explanations for non-linear classier decisions by layer-wise relevance propagation. PLOS ONE 10 (7), 1-46.

#### Section 2.4

[5] Montavon, G., Lapuschkin, S., Binder, A., Samek, W., Müller, K.R., 2017. Explaining nonlinear classication decisions with deep Taylor decomposition. Pattern Recognition 65, 211-222.

#### Section 2.5

[6] Avanti Shrikumar, Peyton Greenside, and Anshul Kundaje. Learning Important Features Through Propagating Activation Differences. arXiv preprint arXiv:1704.02685, 2017.

#### Section 3.1

[7] Zeiler, M. D., Fergus, R., 2014. Visualizing and understanding convolutional networks. In: Computer Vision - ECCV 2014 - 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part I. pp. 818-833.

#### Section 3.2

[8] K. Simonyan, A. Vedaldi, and A. Zisserman. Deep inside convolutional networks: Visualising image classification models and saliency maps. In Workshop at International Conference on Learning Representations, 2014.

#### Section 3.3

[9] Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, and Martin Riedmiller. Striving for simplicity: The all convolutional net. arXiv preprint arXiv:1412.6806, 2014.

#### Section 3.4

[10] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. arXiv preprint arXiv:1703.01365, 2017.

#### Section 3.5

[11] Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, and Martin Wattenberg. SmoothGrad: removing noise by adding noise. arXiv preprint arXiv:1706.03825, 2017.

#### Section 4.1

[12] Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba. Learning deep features for discriminative localization. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2921–2929, 2016.

#### Section 4.2

[13] R. R.Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, and D. Batra. Grad-cam: Why did you say that? visual explanations from deep networks via gradient-based localization. arXiv:1611.01646, 2016.

#### Section 4.3

[14] A. Chattopadhyay, A. Sarkar, P. Howlader, and V. N. Balasubramanian. Grad-cam++: Generalized gradient-based visual explanations for deep convolutional networks. CoRR, abs/1710.11063, 2017.
