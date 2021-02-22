# Contrastive-Learning-in-ESC
<h3> Three Training Schemes For The Audio Classifiers: </h3>


<p align="center"><img src="resources/overview.jpg" alt="overview of our three models" height="900"></p>

<h3> Data Augmentation: </h3>
<img src="resources/data_augmentation.jpg" alt="data augmentation process">

### Setup and Dependencies:
First, install the above dependencies.

Second, download ESC50 and US8K datasets and put them inside the 'data' directory, and assign True to the dataset that you want to use to train your model, in config.py

###Quickstart
1. To train a classifier with ResNet-50 with cross-entropy loss:
```
$ python train_crossEntropyLoss.py
```
To train a classifier with ResNet-50 with supervised-contrastive loss:
Stage 1:
```
$ python train_contrastiveLoss.py
```
Stage2:
and then update path_to_classifierModel in config.py with the path to the ResNet model trained in stage1, and:
```
$ python train_classifier.py
```

To train a classifier with ResNet-50 with hybrid loss:
```
$ python train_hybridLoss.py
```

###Changing Hyperparameters
You can change the 
