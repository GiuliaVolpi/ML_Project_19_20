# Machine Learning Project 2019/2020

## Abastract
In this work, we developed a new Neural Network library in Python, and we called it IsaNet Lib. It provides high and low-level API with an entire module for model selection. First, it was tested on the MONK Datasets (classification tasks) and then on the CUP Dataset (regression task). All the analyses use well-known validation methods (Hold-out, K-fold Cross Validation) to provide a model that can generalize the data.

## Introduction
The aims of project A of the Machine Learning (ML) course are to implement an ML model simulator (Neural Network, SVM, ...), understand the hyper-parameters effect on the model and solve a supervised regression learning task by using the CUP dataset provided in the course. For the first aim, we decided to implement a Multi-Layer Perceptron (MLP) model with back-propagation (gradient descent), Classic Momentum (also known as Polyak’s heavy ball method[1]), Nesterov (introduced in [2]), the new Accelerated Nesterov (see [3]), L2 regularization, some Early Stopping, and other features. Since the project grows very fast, we decided to develop a neural network library that would allow us to focus on the other aims. We will explain all the major details of the library in the next session. In order to test and validate our library we used MONK datasets [4] and after that, we moved on to CUP dataset analysis. To determine the best hyper-parameters configuration for the CUP’s task we performed a grid search with K-fold cross-validation and the final model was re-trained using hold out as validation method for the early stopping.

All the detalis about the project can be found on the full report [here](https://github.com/alessandrocuda/ML_Project_19_20/blob/master/report/isanet_report_ML_19.pdf).

## References
 - [1] Boris Polyak. Some methods of speeding up the convergence of iteration methods. Ussr Computational Mathematics and Mathematical Physics, 4:1–17, 12 1964.
 - [2] I. Sutskever, J. Martens, G. Dahl, and G. Hinton. On the importance of initialization and momentum in deep learning. 30th International Conference on Machine Learning, ICML 2013, pages 1139–1147, 01 2013.
 - [3] Goran Nakerst, John Brennan, and Masudul Haque. Gradient descent with momentum — to accelerate or to super-accelerate?, 2020.
 - [4] Dheeru Dua and Casey Graff. UCI machine learning repository, 2017.
 - [5] Intel Math Kernel Library. Reference Manual. Intel Corporation, Santa Clara, USA, 2009. ISBN 630813-054US.
 - [6] LutzPrechelt.EarlyStopping—ButWhen?,pages53–67.SpringerBerlinHeidelberg,Berlin,Heidelberg, 2012.
 - [7] Simon S. Haykin. Neural networks and learning machines. Pearson Education, Upper Saddle River, NJ, third edition, 2009.
 - [8] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems, page 2012.
 - [9] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS’10). Society for Artificial Intelligence and Statistics, 2010.