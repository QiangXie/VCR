import os
import sys
from data import DataIterator
import config
import tensorflow as tf
from model import model
from model import timer
import datetime
import operator
import numpy as np

FLAGS = config.FLAGS

class Trainer(object):
    def __init__(self, train_data_dir=None, val_data_dir=None, device="/cpu:0"):
        self.log_dir = os.path.join(FLAGS.log_dir, "train")
        if os.path.exists(self.log_dir):
            log_files = os.listdir(self.log_dir)
            for log_file in log_files:
                os.remove(os.path.join(self.log_dir, log_file))
        self.device = device
        print("Loading train data, please wait---------")
        with tf.device(self.device):
            self.train_feeder = DataIterator.DataIterator2(FLAGS.train_data_dir)
            self.train_batch_inputs, self.train_batch_sparse_labels, self.train_batch_encode_labels =\
                    self.train_feeder.feed_tensor()
            print("Load train data done, load {} images.".format(self.train_feeder.size))
            print("Loading val data, please wait---------")
            self.val_feeder = DataIterator.DataIterator2(FLAGS.val_data_dir)
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
            self.vin_rec_model = model.LSTMOCR('train', batch_inputs, batch_sparse_labels, batch_encode_labels)
            self.train_num_batches_per_epoch = int(self.train_feeder.size / FLAGS.batch_size)
            self.val_num_batches_per_epoch = int(self.val_feeder.size / FLAGS.batch_size)
            self.vin_rec_model.build_graph(self.train_num_batches_per_epoch)
        with tf.device("/cpu:0"):
            config = tf.ConfigProto(allow_soft_placement=True)
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            all_variables_list = tf.global_variables()
            #train step from 0 
            restore_variables_list = []
            if FLAGS.step_set0:
                for item in all_variables_list:
                    if item.name != "global_step:0":
                        restore_variables_list.append(item)
            else:
                restore_variables_list = all_variables_list
            self.saver = tf.train.Saver(restore_variables_list, max_to_keep=100)
            self.tb_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', self.sess.graph)
            if FLAGS.restore:
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                if ckpt:
                    self.saver.restore(self.sess, ckpt)
                    print("Restore from checkpoint {}".format(ckpt))
            self.coord = tf.train.Coordinator()
            self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
            self.timer = timer.Timer()

    def train(self):
        print("-------------------beging training---------------------------")
        with tf.device("/cpu:0"):
            for cur_epoch in range(FLAGS.num_epochs):
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
                    train_cost += cost_averge * FLAGS.batch_size
                    self.tb_writer.add_summary(summary_str, step)
                    if (step + 1) % FLAGS.save_step_interval == 0:
                        self.save_ckpt(step)
                    if (step + 1) % FLAGS.validation_interval_steps == 0:
                        self.val(cur_epoch)
        print("-------------------train done---------------------------------")


    def save_ckpt(self, step):
        if not os.path.exists(FLAGS.checkpoint_dir):
            os.mkdir(FLAGS.checkpoint_dir)
        print("Save the checkpoint of {}".format(step))
        self.saver.save(self.sess, os.path.join(FLAGS.checkpoint_dir, 'vin-rec-model'),
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
            FLAGS.num_epochs, cur_epoch, accuracy))











        
