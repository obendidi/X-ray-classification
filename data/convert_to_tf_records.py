import random
import tensorflow as tf
from dataset_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset



#===============DEFINE YOUR ARGUMENTS==============
flags = tf.app.flags

#State your dataset file
flags.DEFINE_string('dataset_file', "data.txt", 'String: Your dataset txt file')

# The number of shards per dataset split.
flags.DEFINE_integer('num_shards', 2, 'Int: Number of shards to split the TFRecord files')

# The number of images in the validation set. You would have to know the total number of examples in advance. This is essentially your evaluation dataset.
flags.DEFINE_float('validation_size', 0.3, 'Float: The proportion of examples in the dataset to be used for validation')
# Seed for repeatability.
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

#Output filename for the naming the TFRecord file
flags.DEFINE_string('tfrecord_filename', "X_ray", 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS

def main():

    #=============CHECKS==============
    #Check if there is a tfrecord_filename entered
    if not FLAGS.tfrecord_filename:
        raise ValueError('tfrecord_filename is empty. Please state a tfrecord_filename argument.')

    #If the TFRecord files already exist in the directory, then exit without creating the files again
    if _dataset_exists(dataset_file = FLAGS.dataset_file, _NUM_SHARDS = FLAGS.num_shards, output_filename = FLAGS.tfrecord_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return None
    #==========END OF CHECKS============

    #Get a list of photo_filenames like ['123.jpg', '456.jpg'...] and a list of sorted class names from parsing the subdirectories.
    photo_filenames, class_names = _get_filenames_and_classes(FLAGS.dataset_file)

    #Refer each of the class name to a specific integer number for predictions later
    class_ids = [1 if label == "normal" else 0 for label in class_names]

    #Find the number of validation examples we need
    num_validation = int(FLAGS.validation_size * len(photo_filenames))

    training_filenames = photo_filenames[num_validation:]
    validation_filenames = photo_filenames[:num_validation]
    training_labels = class_ids[num_validation:]
    validation_labels = class_ids[:num_validation]
    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, training_labels,
                     dataset_file = FLAGS.dataset_file,
                     tfrecord_filename = FLAGS.tfrecord_filename,
                     _NUM_SHARDS = FLAGS.num_shards)
    _convert_dataset('validation', validation_filenames, validation_labels,
                     dataset_file = FLAGS.dataset_file,
                     tfrecord_filename = FLAGS.tfrecord_filename,
                     _NUM_SHARDS = FLAGS.num_shards)

    print('\nFinished converting the %s dataset!' % (FLAGS.tfrecord_filename))

if __name__ == "__main__":
    main()
