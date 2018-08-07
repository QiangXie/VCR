import os
import sys
from data import DataIterator
import tensorflow as tf
from model import model
from model import timer
import datetime
import operator
import numpy as np
import shutil

class Trainer(object):
    def __init__(self, train_data_dir=None, val_data_dir=None, log_dir=None,
            batch_size=None, step_set0=None, restore=False, checkpoint_dir=None,
            train_epochs=None, save_step_interval=None, validation_interval_steps = None,
            image_height=None, image_width=None, image_channel=None,
            out_channels=None,
            leakiness=None,
            num_hidden=None,
            output_keep_prob=None,
            num_classes=None,
            initial_learning_rate=1e-3,
            decay_epoch=50,
            decay_rate=0.1,
            beta1=None, 
            beta2=None, 
            device="/cpu:0",
            data_type=2,
            augment_factor=20):
        self.restore = restore
        self.step_set0 = step_set0
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.train_epochs = train_epochs
        self.save_step_interval = save_step_interval
        self.validation_interval_steps = validation_interval_steps

        self.train_data_dir = train_data_dir
        if not os.path.exists(self.train_data_dir):
            raise KeyError("train data path don't exists'")
        self.val_data_dir = val_data_dir
        if not os.path.exists(self.val_data_dir):
            raise KeyError("val data path don't exists'")

        self.log_dir = os.path.join(log_dir, "train")
        if os.path.exists(self.log_dir):
            log_files = os.listdir(self.log_dir)
            for log_file in log_files:
                shutil.rmtree(os.path.join(self.log_dir, log_file))
        self.device = device
        print("Loading train data, please wait---------")
        with tf.device(self.device):
            if data_type == 2:
                self.train_feeder = DataIterator.DataIterator2(self.train_data_dir,
                        image_width, image_height, image_channel, self.batch_size, augment_factor)
            self.train_batch_inputs, self.train_batch_sparse_labels, self.train_batch_encode_labels =\
                    self.train_feeder.feed_tensor()
            print("Load train data done, load {} images.".format(self.train_feeder.size))
            print("Loading val data, please wait---------")
            if data_type == 2:
                self.val_feeder = DataIterator.DataIterator2(self.val_data_dir, 
                        image_width, image_height, image_channel, self.batch_size, augment_factor)
            self.val_batch_inputs, self.val_batch_sparse_labels, self.val_batch_encode_labels =\
                    self.val_feeder.feed_tensor()
            print("Load val data done, load {} images.".format(self.val_feeder.size))

            self.train_val_flag = tf.placeholder(dtype=bool, shape=())
            batch_inputs, batch_sparse_labels, batch_encode_labels =\
                    tf.cond(self.train_val_flag,
                            lambda:[self.train_batch_inputs, 
                                self.train_batch_sparse_labels, self.train_batch_encode_labels],
                            lambda:[self.val_batch_inputs, 
                                self.val_batch_sparse_labels, self.val_batch_encode_labels])
            self.vin_rec_model = model.LSTMOCR('train', batch_inputs, batch_sparse_labels, batch_encode_labels,
                    image_height, image_width, image_channel,
                    out_channels, leakiness, num_hidden,
                    output_keep_prob, num_classes)
            self.train_num_batches_per_epoch = int(self.train_feeder.size / self.batch_size)
            self.val_num_batches_per_epoch = int(self.val_feeder.size / self.batch_size)
            self.vin_rec_model.build_graph(self.train_num_batches_per_epoch, 
                    initial_learning_rate, 
                    decay_epoch,
                    decay_rate,
                    beta1,
                    beta2,
                    batch_size)
        with tf.device("/cpu:0"):
            config = tf.ConfigProto(allow_soft_placement=True)
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            all_variables_list = tf.global_variables()
            #train step from 0 
            restore_variables_list = []
            if self.step_set0:
                for item in all_variables_list:
                    if item.name != "global_step:0":
                        restore_variables_list.append(item)
            else:
                restore_variables_list = all_variables_list
            self.saver = tf.train.Saver(restore_variables_list, max_to_keep=100)
            self.tb_writer = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
            if self.restore:
                ckpt = tf.train.latest_checkpoint(self.checkpoint_dir)
                if ckpt:
                    self.saver.restore(self.sess, ckpt)
                    print("Restore from checkpoint {}".format(ckpt))
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
            self.timer = timer.Timer()

    def train(self):
        print("-------------------beging training---------------------------")
        with tf.device("/cpu:0"):
            for cur_epoch in range(self.train_epochs):
                print("Epoch {}...".format(cur_epoch))
                train_cost = 0

                for cur_batch in range(self.train_num_batches_per_epoch):
                    self.timer.tic()
                    feed = {self.train_val_flag: True}
                    summary_str, cost_averge, step, _ , lr, rec_result, label=\
                            self.sess.run([self.vin_rec_model.merged_summay, 
                                self.vin_rec_model.cost, 
                                self.vin_rec_model.global_step,
                                self.vin_rec_model.train_op, 
                                self.vin_rec_model.lrn_rate,
                                self.vin_rec_model.dense_decoded,
                                self.train_batch_encode_labels],
                                feed_dict=feed)
                    
                    if (cur_batch + 1) % 20 == 0:
                        self.timer.toc()
                        print("Batch {} time consuming: {:.5f}s, last_batch_err = {:.5f}, lr = {:.5f}".
                                format(cur_batch, self.timer.total_time, cost_averge, lr))
                    train_cost += cost_averge * self.batch_size
                    self.tb_writer.add_summary(summary_str, step)
                    if (step + 1) % self.save_step_interval == 0:
                        self.save_ckpt(step)
                    if (step + 1) % self.validation_interval_steps == 0:
                        self.val(cur_epoch)
        print("-------------------train done---------------------------------")


    def save_ckpt(self, step):
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        print("Save the checkpoint of {}".format(step))
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'vin-rec-model'),
                global_step = step)
            
    def val(self, cur_epoch):
        accuracy = 0.0
        total = 0
        count = 0
        for i in range(self.val_num_batches_per_epoch):
            val_feed = {self.train_val_flag: False}
            labels, prediction= self.sess.run([self.vin_rec_model.encode_labels,
                                               self.vin_rec_model.dense_decoded],
                                               feed_dict=val_feed)
            if len(labels) != len(prediction):
                raise Exception("Length of labels do not equal length of prediction!")
            for i in range(len(labels)):
                total += 1
                if np.all(labels[i] == prediction[i]):
                    count += 1
        accuracy = float(count) / float(total)
        now = datetime.datetime.now()
        print("{}/{} {}:{}:{} Epoch {}/{} Accuracy = {:.5f}".format(now.month, now.day, now.hour, now.minute, now.second,
            self.train_epochs, cur_epoch, accuracy))


