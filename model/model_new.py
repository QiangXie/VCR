import tensorflow as tf
import sys 
import config
from tensorflow.python.training import moving_averages

FLAGS = config.FLAGS

class LSTMOCR(object):
    def __init__(self, mode, inputs = None, sparse_labels = None, encode_labels=None):
        self.mode = mode
        if self.mode == "train":
            self.inputs = inputs
            self.sparse_labels = sparse_labels
            self.encode_labels = encode_labels
            #self.seq_len = seq_len
        else:
            self.inputs = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
            self.sparse_labels = tf.sparse_placeholder(tf.int32)
            self.seq_len = tf.placeholder(tf.int32, [None])
        self._extra_train_ops = list()

    def build_graph(self):
        self._build_model()
        self._build_train_op()
        self.merged_sumary = tf.summary.merge_all()

    def _build_model(self):
        feature_map_channels = [[FLAGS.image_channel, 64],
                                [64, 128],
                                [128, 128],
                                [128, FLAGS.out_channels]]
        strides = [1, 2]

        x = self.inputs
        with tf.variable_scope('CNN'):
            for i_ in range(len(feature_map_channels)):
                with tf.variable_scope('Unit-{}'.format(i_+1)):
                    for i__ in range(len(feature_map_channels[i_])-1):
                        x = self._conv2d(x, "cnn-{}-{}".format(i_+1, i__+1), 
                                3, feature_map_channels[i_][i__], feature_map_channels[i_][i__+1],
                                strides[0])
                        x = self._batch_norm('bn{}-{}'.format(i_+1, i__+1), x)
                        x = self._leaky_relu(x, 0.01)
                    x = self._max_pool(x, 2, strides[1])

        print("CNN feature shape:{}.".format(x.shape))
        feature_b, feature_h, feature_w, feature_c = x.get_shape().as_list()

        with tf.variable_scope('Reshape'):
            
            #trans to [batch_size, feature_w, feature_h, out_channels]
            x = tf.transpose(x, [0, 2, 1, 3])
            #batch_size * 48 * 1024 
            x = tf.reshape(x, [feature_b, feature_w, feature_h*feature_c]) 
            #exit()
            
            #x = tf.reshape(x, [FLAGS.batch_size, -1, feature_map_channels[12]])
            #x = tf.transpose(x, [0, 2, 1])
            #x.set_shape([FLAGS.batch_size, feature_map_channels[12], feature_w*feature_h])

            #x = self._dense(x, FLAGS.batch_size, feature_h*feature_w, feature_w)
            #x = self._dense(x, FLAGS.batch_size, feature_h*FLAGS.out_channels, 128)
            #x = self._batch_norm("dense-norm", x)
            #x = self._leaky_relu(x, 0.01)

        with tf.variable_scope('LSTM'):
            self.seq_len = tf.fill([x.get_shape().as_list()[0]], feature_w)
            print("seq_len shape: {}".format(self.seq_len.shape))

            '''cell_1_fw = tf.nn.rnn_cell.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            if self.mode == "train":
                cell_1_fw = tf.nn.rnn_cell.DropoutWrapper(cell=cell_1_fw, output_keep_prob=0.8)

            cell_1_bw = tf.nn.rnn_cell.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            if self.mode == "train":
                cell_1_bw = tf.nn.rnn_cell.DropoutWrapper(cell=cell_1_bw, output_keep_prob=0.8)

            cell_2_fw = tf.nn.rnn_cell.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            if self.mode == "train":
                cell_2_fw = tf.nn.rnn_cell.DropoutWrapper(cell=cell_2_fw, output_keep_prob=0.8)
            
            cell_2_bw = tf.nn.rnn_cell.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)
            if self.mode == "train":
                cell_2_bw = tf.nn.rnn_cell.DropoutWrapper(cell=cell_2_bw, output_keep_prob=0.8)

            cells = [cell1, cell2]
            #stack rnn cells
            stack = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
            initial_state = stack.zero_state(FLAGS.batch_size, dtype = tf.float32)
            outputs, _ = tf.nn.dynamic_rnn(
                    cell=stack, 
                    inputs=x, 
                    sequence_length=self.seq_len, 
                    initial_state=initial_state,
                    dtype=tf.float32,
                    time_major=False)
            '''
            outputs = self._stacked_bidirectional_rnn(RNN=tf.nn.rnn_cell.GRUCell, 
                    num_units=FLAGS.num_hidden, 
                    num_layers=2,
                    inputs=x,
                    seq_lengths=self.seq_len,
                    batch_size=FLAGS.batch_size)

            outputs = tf.reshape(outputs, [-1, FLAGS.num_hidden])

            W = tf.get_variable(name='W',
                    shape=[FLAGS.num_hidden, FLAGS.char_classes],
                    dtype=tf.float32,
                    initializer=tf.glorot_uniform_initializer())
            b = tf.get_variable(name='b',
                    shape=[FLAGS.char_classes],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer())
            
            self.logits = tf.matmul(outputs, W) + b
            shape = tf.shape(x)
            self.logits = tf.reshape(self.logits, [shape[0], -1, FLAGS.char_classes])
            self.logits = tf.transpose(self.logits, (1, 0, 2))

    def _stacked_bidirectional_rnn(self, RNN, num_units, num_layers, inputs, seq_lengths, batch_size):
        _inputs = inputs
        if len(_inputs.get_shape().as_list()) != 3:
            raise ValueError("inputs mus be 3-dimentional tensor")
        for _ in range(num_layers):
            with tf.variable_scope(None, default_name="bidirectionsa-rnn"):
                rnn_cell_fw = RNN(num_units)
                if self.mode == "train":
                    rnn_cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell=rnn_cell_fw, output_keep_prob=0.8)
                rnn_cell_bw = RNN(num_units)
                if self.mode == "train":
                    rnn_cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell=rnn_cell_bw, output_keep_prob=0.8)
                initial_state_fw = rnn_cell_fw.zero_state(batch_size, dtype=tf.float32)
                initial_state_bw = rnn_cell_bw.zero_state(batch_size, dtype=tf.float32)
                (output, state) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, _inputs, seq_lengths,
                        initial_state_fw, initial_state_bw, dtype=tf.float32)
                _inputs = tf.concat(output, 2)
        return _inputs
        


    def _dense(self, x, batch_size, in_size, out_size):
        x = tf.reshape(x, [-1, in_size])
        W = tf.get_variable(name="W", 
                shape=[in_size, out_size],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name='b',
                shape=[out_size],
                dtype=tf.float32,
                initializer=tf.constant_initializer())
        out = tf.matmul(x, W) + b
        out = tf.reshape(out, [batch_size, -1, out_size])
        return out


    def _build_train_op(self):
        #self.global_step = tf.Variable(0, trainable=False)
        self.global_step = tf.train.get_or_create_global_step()
        self.loss = tf.nn.ctc_loss(labels=self.sparse_labels,
                inputs=self.logits,
                sequence_length=self.seq_len)
        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar('cost', self.cost)
        self.lrn_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                self.global_step,
                FLAGS.decay_steps,
                FLAGS.decay_rate,
                staircase=True)
        tf.summary.scalar('lr', self.lrn_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lrn_rate,
                beta1=FLAGS.beta1, beta2=FLAGS.beta2).minimize(self.loss,
                        global_step=self.global_step)
        train_ops = [self.optimizer] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)
        self.ctc_decoded_labels_sparse, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits,
                self.seq_len,
                merge_repeated=False)
        self.ctc_decoded_labels = tf.sparse_tensor_to_dense(self.ctc_decoded_labels_sparse[0], default_value=-1)
        self.accuracy_batch, _ = tf.metrics.accuracy(self.encode_labels, self.ctc_decoded_labels)

    def _conv2d(self, x, name, filter_size, in_channels, out_channels, strides):
        with tf.variable_scope(name):
            kernel = tf.get_variable(name="DW", 
                    shape=[filter_size, filter_size, in_channels, out_channels],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer())
            b = tf.get_variable(name='bias', 
                    shape=[out_channels],
                    dtype=tf.float32,
                    initializer=tf.constant_initializer())
            conv2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding="SAME")

        return tf.nn.bias_add(conv2d_op, b)

    def _batch_norm(self, name, x):
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable(
                    'beta', params_shape, tf.float32, 
                    initializer=tf.constant_initializer(0.0, tf.float32))
            gama = tf.get_variable(
                    'gama', params_shape, tf.float32,
                    initializer=tf.constant_initializer(1.0, tf.float32))
            if self.mode == "train":
                mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
                moving_mean = tf.get_variable(
                        'moving_mean', params_shape, tf.float32,
                        initializer=tf.constant_initializer(0.0, tf.float32),
                        trainable=False)
                moving_variance = tf.get_variable(
                        'moving_variance', params_shape, tf.float32,
                        initializer=tf.constant_initializer(1.0, tf.float32),
                        trainable = False
                        )

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_mean, mean, 0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    moving_variance, variance, 0.9))
            else:
                mean = tf.get_variable(
                        'moving_mean', params_shape, tf.float32,
                        initializer=tf.constant_initializer(0.0, tf.float32),
                        trainable=False)
                variance = tf.get_variable(
                        'moving_mean', params_shape, tf.float32,
                        initializer=tf.constant_initializer(1.0, tf.float32),
                        trainable=False)

                tf.summary.histogram(mean.op.name, mean)
                tf.summary,histogram(variance.op.name, variance)

            x_bn = tf.nn.batch_normalization(x, mean, variance, beta, gama, 0.001)
            x_bn.set_shape(x.get_shape())

            return x_bn

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.nn.relu(x)
        #return tf.where(tf.less(x, 0.0), leakiness*x, x, name="leaky_relu")

    def _max_pool(self, x, ksize, strides):
        return tf.nn.max_pool(x, 
                ksize=[1, ksize, ksize, 1],
                strides=[1, strides, strides, 1],
                padding="SAME",
                name='max_pool')
        
