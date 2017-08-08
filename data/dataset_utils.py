import math
import os
import sys
import tensorflow as tf

slim = tf.contrib.slim

#State the labels filename
LABELS_FILENAME = 'labels.txt'
#===================================================  Dataset Utils  ===================================================

def int64_feature(values):
  """Returns a TF-Feature of int64s.
  Args:
    values: A scalar or list of values.
  Returns:
    a TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.
  Args:
    values: A string.
  Returns:
    a TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))

def write_label_file(labels_to_class_names, dataset_file,
                     filename=LABELS_FILENAME):
  """Writes a file with the list of class names.
  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_file: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  labels_filename = os.path.join(dataset_file, filename)
  with tf.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_file, filename=LABELS_FILENAME):
  """Specifies whether or not the dataset directory contains a label map file.
  Args:
    dataset_file: The directory in which the labels file is found.
    filename: The filename where the class names are written.
  Returns:
    `True` if the labels file exists and `False` otherwise.
  """
  return tf.gfile.Exists(os.path.join(dataset_file, filename))


def read_label_file(dataset_file, filename=LABELS_FILENAME):
  """Reads the labels file and returns a mapping from ID to class name.
  Args:
    dataset_file: The directory in which the labels file is found.
    filename: The filename where the class names are written.
  Returns:
    A map from a label (integer) to class name.
  """
  labels_filename = os.path.join(dataset_file, filename)
  with tf.gfile.Open(labels_filename, 'r') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    labels_to_class_names[int(line[:index])] = line[index+1:]
  return labels_to_class_names

#=======================================  Conversion Utils  ===================================================

#Create an image reader object for easy reading of the images
class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB png data.
    self._decode_png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_png(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_png(self, sess, image_data):
    image = sess.run(self._decode_png,
                     feed_dict={self._decode_png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_file):
  """Returns a list of filenames and inferred class names.
  Args:
    dataset_file: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or png encoded images.
  Returns:
    A list of image file paths, relative to `dataset_file` and the list of
    subdirectories, representing class names.
  """
  files = [line.rstrip() for line in open(dataset_file)]
  photo_filenames = [filename.split(" ")[1] for filename in files]
  class_names = [filename.split(" ")[0] for filename in files]

  return photo_filenames,class_names


def _get_dataset_filename(dataset_file, split_name, shard_id, tfrecord_filename, _NUM_SHARDS):
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
      tfrecord_filename, split_name, shard_id, _NUM_SHARDS)
  return output_filename


def _convert_dataset(split_name, filenames, class_ids, dataset_file, tfrecord_filename, _NUM_SHARDS):
  """Converts the given filenames to a TFRecord dataset.
  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or png images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_file: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_file, split_name, shard_id, tfrecord_filename = tfrecord_filename, _NUM_SHARDS = _NUM_SHARDS)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_id = class_ids[i]

            example = image_to_tfexample(
                image_data, b'png', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()

def _dataset_exists(dataset_file, _NUM_SHARDS, output_filename):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      tfrecord_filename = _get_dataset_filename(
          dataset_file, split_name, shard_id, output_filename, _NUM_SHARDS)
      if not tf.gfile.Exists(tfrecord_filename):
        return False
  return True
