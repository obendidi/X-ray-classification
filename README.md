# X-ray-classification

Exploiting transfer learning methods to try and classify X-ray chest Images into normal(healthy) **vs** abnormal(sick)

we will see the performance of transfer learning using the official pre-trained model offered by Google (INCEPTION-RESNET-V2 MODEL), which can be found in TensorFlow’s model library

In this little/first try we will be retraining the last layer of inception v2 of google to classify the images using adam optimizer and learning rate decay

## Sample dataset

<img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/28.png" width="260" height="260"/> <img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/29.png" width="260" height="260" /> <img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/30.png" width="260" height="260" />
<img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/31.png" width="260" height="260" /> <img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/32.png" width="260" height="260" /> <img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/33.png" width="260" height="260" />


## Requirements

  python 3
  tensorflow = 1.0.1
  matplotlib
  lxml

## Training Specification

**model used** : INCEPTION-RESNET-V2

**training layers** : Last layer only

**learning rate** : 0.0001 with a decay factor of 0.7 each 2 epochs

**batch size** : 16

**number of epochs** : 30


## Results on test set

**Streaming Accuracy** : *68.70 %*
<img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/validation_accuracy.png" width="800" height="400" />


**Recall** : *coming soon*

**Precision** : *coming soon*

## Sample Predictions

<img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/figure_1-16.png" width="300" height="300"/> <img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/figure_1-2.png" width="300" height="300" /> <img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/figure_1-5.png" width="300" height="300" />
<img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/figure_191.png" width="300" height="300" /> <img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/figure_1-3.png" width="300" height="300" /> <img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/figure_51-1.png" width="300" height="300" />
