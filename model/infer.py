import os
import tensorflow as tf
from model import model
import cv2
import numpy as np
from data import DataIterator

class Infer(object):
    def __init__(self, ckpt_path=None, 
            width=384, height=64, channel=3, batch_size=1,
            out_channels=64, leakiness=0.01, num_hidden=128, num_classes=35,
            gpu_id = None):
        if not os.path.exists(ckpt_path):
            raise ValueError("check point path don't exist")
        if os.path.isdir(ckpt_path):
            self.ckpt = tf.train.latest_checkpoint(ckpt_path) 
        else:
            self.ckpt = ckpt_path

        self.gpu = gpu_id
        if self.gpu == None:
            self.device = "/cpu:0"
        else:
            self.device = "/gpu:{}".format(self.gpu)
        self.width = width
        self.height = height
        self.channel = channel
        self.out_channels = out_channels
        self.leakiness = leakiness
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size

    def build_model(self):
        with tf.device(self.device):
            self.vin_rec_model = model.LSTMOCR(mode='infer',
                    image_height=self.height,
                    image_width=self.width,
                    image_channel=self.channel,
                    out_channels=self.out_channels,
                    leakiness=self.leakiness,
                    num_hidden=self.num_hidden,
                    output_keep_prob=None,
                    num_classes=self.num_classes
                    )
            self.vin_rec_model.build_graph(batch_size=self.batch_size)
        with tf.device("/cpu:0"):
            config = tf.ConfigProto(allow_soft_placement=True)
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, self.ckpt)
            self.build_preprocess_tf()
    
    def recognize_from_path(self, path):
        feed = {self.image_path_tensor:path}
        inputs = self.sess.run(self.processed_image, feed)
        feed = {self.vin_rec_model.inputs: inputs}
        out = self.sess.run(self.vin_rec_model.dense_decoded, feed)
        decode_outs = []
        for item in out:
            decode_out = ""
            for i in item:
                if i == -1:
                    decode_out += ""
                else:
                    decode_out += DataIterator.decode_maps[i]
            decode_outs.append(decode_out)
        return decode_outs


    def preprocess_opencv(self, path):
        im = cv2.imread(path)
        im = cv2.resize(im, (self.width, self.height), cv2.INTER_LINEAR).astype(np.float32)*1.0/127.5 - 1.0
        im = np.reshape(im, [self.height, self.width, self.channel])
        img_input = [im]
        img_input = np.asarray(img_input)
        return img_input

    def build_preprocess_tf(self):
        self.image_path_tensor = tf.placeholder(tf.string)
        image = tf.read_file(self.image_path_tensor)
        image = tf.image.decode_jpeg(image, channels=self.channel, dct_method="INTEGER_ACCURATE")
        image = tf.image.resize_images(image, [self.height, self.width], method=tf.image.ResizeMethod.BILINEAR)
        image  = image*1.0/127.5 - 1.0
        self.processed_image = tf.expand_dims(image, 0)


