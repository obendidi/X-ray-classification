# X-ray-classification

Exploiting transfer learning methods to try and classify X-ray chest Images into normal(healthy) **vs** abnormal(sick)

we will see the performance of transfer learning using the official pre-trained model offered by Google (INCEPTION-RESNET-V2 MODEL), which can be found in TensorFlow’s model library

In this little/first try we will be retraining the last layer of inception v2 of google to classify the images using adam optimizer and learning rate decay

## Image example

<img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/28.png" width="250" height="250"/> <img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/29.png" width="250" height="250" /> <img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/30.png" width="250" height="250" />
<img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/31.png" width="250" height="250" /> <img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/32.png" width="250" height="250" /> <img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/33.png" width="250" height="250" />
