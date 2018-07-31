from data import VinAnnotationsReader
import tensorflow as tf
import numpy as np
import sys
import config
import os
from tqdm import tqdm

FLAGS = config.FLAGS

charset = u'0123456789ABCDEFGHJKLMNPRSTUVWXYZ'
#charset = u'0123456789+-*()'
encode_maps = {}
decode_maps = {}

for i, char in enumerate(charset, 1):
    encode_maps[char] = i
    decode_maps[i] = char

SPACE_INDEX = 0
SPACE_TOKEN = ''
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN

class DataIterator(object):
    def __init__(self, data_dir):
        vin_annotations_reader = VinAnnotationsReader.VinAnnotationsReader(data_dir)
        self.img_pathes = vin_annotations_reader.get_img_list()
        self.crop_windows = list()
        str_labels = vin_annotations_reader.get_vin_codes()
        boundingboxes = vin_annotations_reader.get_boundingboxs()
        self.encode_labels = list()
        for i,str_label in enumerate(str_labels):
            encode_label = [SPACE_INDEX if str_label == SPACE_TOKEN else encode_maps[c] for c in list(str_label)]
            print("Process {},\nstring label: {}, \nencode label: {}".format(
                self.img_pathes[i], str_label,  encode_label))
            #crop window: [crop_y, crop_x, crop_height, crop_width]
            crop_window = [boundingboxes[i][1], 
                    boundingboxes[i][0], 
                    boundingboxes[i][3] - boundingboxes[i][1], 
                    boundingboxes[i][2] - boundingboxes[i][0]]
            self.encode_labels.append(encode_label)
            self.crop_windows.append(crop_window)

        image_pathes_tensor = tf.convert_to_tensor(self.img_pathes, dtype=tf.string)
        encode_labels_tensor = tf.convert_to_tensor(self.encode_labels, dtype=tf.int32)
        crop_window_tensor = tf.convert_to_tensor(self.crop_windows, dtype=tf.int32)

        image_pathes_queue, encode_labels_queue, crop_window_queue = \
                tf.train.slice_input_producer(
                        [image_pathes_tensor, encode_labels_tensor, crop_window_tensor],
                        shuffle=True)
        image = tf.read_file(image_pathes_queue)
        image = tf.image.decode_and_crop_jpeg(image, crop_window_queue, channels=FLAGS.image_channel)
        image = tf.image.resize_images(image, [FLAGS.image_height, FLAGS.image_width])
        image = image*1.0/127.5 - 1.0

        self.batch_inputs, self.batch_encode_labels = \
                tf.train.batch([image, encode_labels_queue],
                        batch_size = FLAGS.batch_size,
                        capacity = FLAGS.batch_size*8,
                        num_threads = 4)

        indx = tf.where(tf.not_equal(self.batch_encode_labels, 0))
        self.batch_sparse_labels = tf.SparseTensor(indx,
                tf.gather_nd(self.batch_encode_labels, indx),
                self.batch_encode_labels.get_shape())

    @property 
    def size(self):
        return len(self.encode_labels)

    def the_label(self):
        return self.batch_encode_labels

    def feed_tensor(self):
        return self.batch_inputs, self.batch_sparse_labels, self.batch_encode_labels


class DataIterator2(object):
    def __init__(self, data_dir):
        vin_annotations_reader = VinAnnotationsReader.VinAnnotationsReader(data_dir)
        vin_annotations_reader.crop(os.path.join(data_dir, "crop"))
        vin_annotations_reader.data_augmentation(os.path.join(data_dir, "crop"), 
                os.path.join(data_dir, "augment"),
                multiple=FLAGS.augment_factor)
        imgs = os.listdir(os.path.join(data_dir, "augment"))
        self.img_pathes = [os.path.join(os.path.join(data_dir, "augment", img)) for img in imgs]
        str_labels = [os.path.splitext(img)[0].split("_")[1] for img in imgs]
        self.encode_labels = list()
        for i,str_label in enumerate(tqdm(str_labels, ascii=True, desc="Encode labels")):
            #print("Process {}\nstring label: {}\n".format(
            #    self.img_pathes[i], str_label))
            encode_label = [SPACE_INDEX if str_label == SPACE_TOKEN else encode_maps[c] for c in list(str_label)]
            #print("encode label: {}".format(encode_label))
            self.encode_labels.append(encode_label)

        image_pathes_tensor = tf.convert_to_tensor(self.img_pathes, dtype=tf.string)
        encode_labels_tensor = tf.convert_to_tensor(self.encode_labels, dtype=tf.int32)

        image_pathes_queue, encode_labels_queue = \
                tf.train.slice_input_producer(
                        [image_pathes_tensor, encode_labels_tensor],
                        shuffle=True)
        image = tf.read_file(image_pathes_queue)
        image = tf.image.decode_jpeg(image, channels=FLAGS.image_channel)
        image = tf.image.resize_images(image, [FLAGS.image_height, FLAGS.image_width])
        image = image*1.0/127.5 - 1.0

        self.batch_inputs, self.batch_encode_labels = \
                tf.train.batch([image, encode_labels_queue],
                        batch_size = FLAGS.batch_size,
                        capacity = FLAGS.batch_size*8,
                        num_threads = 4)

        indx = tf.where(tf.not_equal(self.batch_encode_labels, 0))
        self.batch_sparse_labels = tf.SparseTensor(indx,
                tf.gather_nd(self.batch_encode_labels, indx),
                self.batch_encode_labels.get_shape())

    @property 
    def size(self):
        return len(self.encode_labels)

    def the_label(self):
        return self.batch_encode_labels

    def feed_tensor(self):
        return self.batch_inputs, self.batch_sparse_labels, self.batch_encode_labels

class DataIterator3(object):
    def __init__(self, data_dir):
        imgs = os.listdir(data_dir)
        self.img_pathes = [os.path.join(data_dir, img) for img in imgs]
        str_labels = [os.path.splitext(img)[0].split("_")[1] for img in imgs]
        self.encode_labels = list()
        for i,str_label in enumerate(str_labels):
            encode_label = [SPACE_INDEX if str_label == SPACE_TOKEN else encode_maps[c] for c in list(str_label)]
            print("Process {},\nstring label: {}, \nencode label: {}".format(
                self.img_pathes[i], str_label,  encode_label))
            self.encode_labels.append(encode_label)

        image_pathes_tensor = tf.convert_to_tensor(self.img_pathes, dtype=tf.string)
        encode_labels_tensor = tf.convert_to_tensor(self.encode_labels, dtype=tf.int32)

        image_pathes_queue, encode_labels_queue = \
                tf.train.slice_input_producer(
                        [image_pathes_tensor, encode_labels_tensor],
                        shuffle=True)
        image = tf.read_file(image_pathes_queue)
        image = tf.image.decode_jpeg(image, channels=FLAGS.image_channel)
        image = tf.image.resize_images(image, [FLAGS.image_height, FLAGS.image_width])
        image = image*1.0/127.5 - 1.0

        self.batch_inputs, self.batch_encode_labels = \
                tf.train.batch([image, encode_labels_queue],
                        batch_size = FLAGS.batch_size,
                        capacity = FLAGS.batch_size*8,
                        num_threads = 4)

        indx = tf.where(tf.not_equal(self.batch_encode_labels, 0))
        self.batch_sparse_labels = tf.SparseTensor(indx,
                tf.gather_nd(self.batch_encode_labels, indx),
                self.batch_encode_labels.get_shape())

    @property 
    def size(self):
        return len(self.encode_labels)

    def the_label(self):
        return self.batch_encode_labels

    def feed_tensor(self):
        return self.batch_inputs, self.batch_sparse_labels, self.batch_encode_labels


class DataIterator4(object):
    def __init__(self, data_dir):
        label_file = open(os.path.join(data_dir, "labels.txt"))
        lines = label_file.readlines()
        self.img_pathes = list()
        str_labels = list()
        for i,line in enumerate(lines):
            label = line.split(' ')[0]
            if len(label) == 7:
                self.img_pathes.append(os.path.join(data_dir, str(i)+".jpg"))
                str_labels.append(label)

        self.encode_labels = list()
        for i,str_label in enumerate(str_labels):
            encode_label = [SPACE_INDEX if str_label == SPACE_TOKEN else encode_maps[c] for c in list(str_label)]
            print("Process {},\nstring label: {}, \nencode label: {}".format(
                self.img_pathes[i], str_label,  encode_label))
            self.encode_labels.append(encode_label)

        image_pathes_tensor = tf.convert_to_tensor(self.img_pathes, dtype=tf.string)
        encode_labels_tensor = tf.convert_to_tensor(self.encode_labels, dtype=tf.int32)

        image_pathes_queue, encode_labels_queue = \
                tf.train.slice_input_producer(
                        [image_pathes_tensor, encode_labels_tensor],
                        shuffle=True)
        image = tf.read_file(image_pathes_queue)
        image = tf.image.decode_jpeg(image, channels=FLAGS.image_channel)
        image = tf.image.resize_images(image, [FLAGS.image_height, FLAGS.image_width])
        image = image*1.0/127.5 - 1.0

        self.batch_inputs, self.batch_encode_labels = \
                tf.train.batch([image, encode_labels_queue],
                        batch_size = FLAGS.batch_size,
                        capacity = FLAGS.batch_size*8,
                        num_threads = 4)

        indx = tf.where(tf.not_equal(self.batch_encode_labels, 0))
        self.batch_sparse_labels = tf.SparseTensor(indx,
                tf.gather_nd(self.batch_encode_labels, indx),
                self.batch_encode_labels.get_shape())

    @property 
    def size(self):
        return len(self.encode_labels)

    def the_label(self):
        return self.batch_encode_labels

    def feed_tensor(self):
        return self.batch_inputs, self.batch_sparse_labels, self.batch_encode_labels
