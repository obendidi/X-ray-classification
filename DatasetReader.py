import numpy as np
import scipy.misc as misc
import os
import logging
import sys


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


class BatchDatset:
    files = []
    images = []
    labels = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, data_file, image_options={}):
        logging.info("Initializing {} Batch Dataset Reader...".format(data_file.split(".")[0]))
        logging.info("Image options are : ")
        print(image_options)
        self.files = [line.rstrip() for line in open(data_file)]
        self.image_options = image_options
        self._read_images()

    def _read_images(self):
        self.__channels = True
        self.images = np.array([self._transform(filename.split(" ")[1]) for filename in self.files])
        self.labels = np.array(
            [1. if filename.split(" ")[0] == "normal" else 0. for filename in self.files])
        logging.info("Images shapes : {}".format(self.images.shape))
        logging.info("Labels shapes : {}".format(self.labels.shape))

    def _transform(self, filename):
        image = misc.imread(filename,mode="RGB")
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            h = int(self.image_options["image_height"])
            w = int(self.image_options["image_width"])
            resize_image = misc.imresize(image,
                                         [h, w], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def get_records(self):
        return self.images, self.labels

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.labels[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.labels[indexes]
