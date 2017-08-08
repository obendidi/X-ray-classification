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

<img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/figure_1-16.png" width="260" height="260"/> <img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/figure_1-2.png" width="260" height="260" /> <img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/figure_1-5.png" width="260" height="260" />
<img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/figure_191.png" width="260" height="260" /> <img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/figure_1-3.png" width="260" height="260" /> <img src="https://github.com/bendidi/X-ray-classification/blob/master/data/sample_imgs/figure_51-1.png" width="260" height="260" />


## Getting started

##### get the data :
In the `data` folder (`cd data/`) :

  1 - Use `python get_data.py` to download scrapped image data from [openi.nlm.nih.gov](https://openi.nlm.nih.gov/gridquery.php?q=&it=x,xg&sub=x&m=1&n=101). It has a large base of Xray,MRI, CT scan images publically available.Specifically Chest Xray Images have been scraped.The images will be downloaded and saved in `images/` and the labels in `data_new.json` (it might take a while)

  Some info about the dataset :
  ```
    Total number of Images : 7469
    The classes with most occurence in the dataset:

    		 ('normal', 2696)
    		 ('No Indexing', 172)
    		 ('Lung/hypoinflation', 88)
    		 ('Thoracic Vertebrae/degenerative/mild', 55)
    		 ('Thoracic Vertebrae/degenerative', 44)
    		 ('Spine/degenerative/mild', 36)
    		 ('Spine/degenerative', 35)
    		 ('Spondylosis/thoracic vertebrae', 33)
    		 ('Granulomatous Disease', 32)
    		 ('Cardiomegaly/mild', 32)
  ```
  2 - Use `python gen_data.py` to sort labels into Normal/Abnormal classes, generate full path to coresponding Images and write them to `data.txt`


  ```
  number of normal chest Images(healthy people) 2696:
  number of abnormal chest Images(sick people) 4773:
  ```

  3 - Use `python convert_to_tf_records.py` to generate tf records of the data.


#### training & evaluation:

  Download the Pre-trained inception model in [here](http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz) and unzip it in `ckpt/` folder.

  Use `python train.py` to start the training !(trained model will be saved in `logs/`)

  Use `python evaluate.py` to run evaluation using the model saved in `logs/`(metric : streaming accuracy over all mini batches)

## References

  [Xvision](https://github.com/ayush1997/Xvision)

  [tensorflow.slim](https://github.com/tensorflow/models/tree/master/slim)

  [tuto.transfer learning](https://kwotsin.github.io/tech/2017/02/11/transfer-learning.html)
