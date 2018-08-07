import tensorflow as tf

class LSTMOCR(object):
    def __init__(self, mode, inputs = None, sparse_labels = None, encode_labels=None,
            image_height=None, image_width=None, image_channel=None,
            out_channels=None,
            leakiness=None, num_hidden=None,
            output_keep_prob=None,
            num_classes=None):
        self.leakiness = leakiness
        self.out_channels = out_channels
        self.image_width = image_width
        self.image_height = image_height
        self.image_channel = image_channel
        self.mode = mode
        self.num_hidden = num_hidden
        self.output_keep_prob = output_keep_prob
        self.num_classes = num_classes
        if self.mode == "train":
            self.inputs = inputs
            self.labels = sparse_labels
            self.encode_labels = encode_labels
            #self.seq_len = seq_len
        else:
            self.inputs = tf.placeholder(tf.float32, [None, self.image_height, self.image_width, self.image_channel])
            self.labels = tf.sparse_placeholder(tf.int32)
            self.seq_len = tf.placeholder(tf.int32, [None])
        self._extra_train_ops = list()

    def build_graph(self, train_num_batches_per_epoch, 
            initial_learning_rate, decay_epoch, decay_rate,
            beta1, beta2, batch_size):
        self._build_model(batch_size)
        self._build_train_op(train_num_batches_per_epoch, initial_learning_rate, 
                decay_epoch, decay_rate,
                beta1, beta2)

        self.merged_summay = tf.summary.merge_all()

    def _build_model(self, batch_size):
        filters = [[self.image_channel, 64], 
                [64, 128], 
                [128, 128],
                [128, 128, self.out_channels]]
        strides = [1, 2]

        feature_h = self.image_height
        feature_w = self.image_width

        #count_ = 0
        #min_size = min(self.image_height, self.image_width)
        #while min_size > 1:
        #    min_size = (min_size + 1) // 2
        #    count_ += 1
        #assert (FLAGS.cnn_count <= count_, "FLAGS.cnn_count should be <= {}!".format(count_))

        # CNN part
        with tf.variable_scope('cnn'):
            x = self.inputs
            for i in range(len(filters)):
                with tf.variable_scope('unit-%d' % (i + 1)):
                    for i_ in range(len(filters[i]) -1 ):
                        x = self._conv2d(x, 'cnn-%d-%d' % (i + 1, i_+1), 3, filters[i][i_], filters[i][i_+1], strides[0])
                        x = self._batch_norm('bn%d-%d' % (i + 1, i_+1), x)
                        x = self._leaky_relu(x, self.leakiness)
                    x = self._max_pool(x, 2, strides[1])

                    # print('----x.get_shape().as_list(): {}'.format(x.get_shape().as_list()))
                    _, feature_h, feature_w, _ = x.get_shape().as_list()
            print('\nfeature_h: {}, feature_w: {}'.format(feature_h, feature_w))

        # LSTM part
        with tf.variable_scope('lstm'):
            x = tf.transpose(x, [0, 2, 1, 3])  # [batch_size, feature_w, feature_h, FLAGS.out_channels]
            # treat `feature_w` as max_timestep in lstm.
            x = tf.reshape(x, [batch_size, feature_w, feature_h * self.out_channels])
            print('lstm input shape: {}'.format(x.get_shape().as_list()))
            self.seq_len = tf.fill([x.get_shape().as_list()[0]], feature_w)
            # print('self.seq_len.shape: {}'.format(self.seq_len.shape.as_list()))

            # tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.GRUCell
            cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden, state_is_tuple=True)
            if self.mode == 'train':
                cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=self.output_keep_prob)

            cell1 = tf.nn.rnn_cell.LSTMCell(self.num_hidden, state_is_tuple=True)
            if self.mode == 'train':
                cell1 = tf.nn.rnn_cell.DropoutWrapper(cell=cell1, output_keep_prob=self.output_keep_prob)

            # Stacking rnn cells
            stack = tf.nn.rnn_cell.MultiRNNCell([cell, cell1], state_is_tuple=True)
            initial_state = stack.zero_state(batch_size, dtype=tf.float32)

            # The second output is the last state and we will not use that
            outputs, _ = tf.nn.dynamic_rnn(
                cell=stack,
                inputs=x,
                sequence_length=self.seq_len,
                initial_state=initial_state,
                dtype=tf.float32,
                time_major=False
            )  # [batch_size, max_stepsize, FLAGS.num_hidden]

            # Reshaping to apply the same weights over the timesteps
            outputs = tf.reshape(outputs, [-1, self.num_hidden])  # [batch_size * max_stepsize, FLAGS.num_hidden]

            W = tf.get_variable(name='W_out',
                                shape=[self.num_hidden, self.num_classes],
                                dtype=tf.float32,
                                initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer
            b = tf.get_variable(name='b_out',
                                shape=[self.num_classes],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            self.logits = tf.matmul(outputs, W) + b
            # Reshaping back to the original shape
            shape = tf.shape(x)
            self.logits = tf.reshape(self.logits, [shape[0], -1, self.num_classes])
            # Time major
            self.logits = tf.transpose(self.logits, (1, 0, 2))

    def _build_train_op(self, train_num_batches_per_epoch, initial_learning_rate, decay_epoch, decay_rate,
            beta1, beta2):
        # self.global_step = tf.Variable(0, trainable=False)
        #self.global_step = tf.train.get_or_create_global_step()
        self.global_step = tf.train.create_global_step()

        self.loss = tf.nn.ctc_loss(labels=self.labels,
                                   inputs=self.logits,
                                   sequence_length=self.seq_len)
        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar('cost', self.cost)

        self.lrn_rate = tf.train.exponential_decay(initial_learning_rate,
                                                   self.global_step,
                                                   decay_epoch*train_num_batches_per_epoch,
                                                   decay_rate,
                                                   staircase=True)
        tf.summary.scalar('learning_rate', self.lrn_rate)

        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lrn_rate,
        #                                            momentum=FLAGS.momentum).minimize(self.cost,
        #                                                                              global_step=self.global_step)
        # self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lrn_rate,
        #                                             momentum=FLAGS.momentum,
        #                                             use_nesterov=True).minimize(self.cost,
        #                                                                         global_step=self.global_step)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lrn_rate,
                                                beta1=beta1,
                                                beta2=beta2).minimize(self.loss,
                                                                            global_step=self.global_step)
        train_ops = [self.optimizer] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)

        # Option 2: tf.nn.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        self.decoded, self.log_prob = \
            tf.nn.ctc_beam_search_decoder(self.logits,
                                          self.seq_len,
                                          merge_repeated=False)
        # self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len,merge_repeated=False)
        self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)
        #self.accuracy_batch, self.accuracy_op = tf.metrics.accuracy(labels=self.encode_labels, predictions=self.dense_decoded)

    def _conv2d(self, x, name, filter_size, in_channels, out_channels, strides):
        with tf.variable_scope(name):
            kernel = tf.get_variable(name='W',
                                     shape=[filter_size, filter_size, in_channels, out_channels],
                                     dtype=tf.float32,
                                     initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer

            b = tf.get_variable(name='b',
                                shape=[out_channels],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME')

        return tf.nn.bias_add(con2d_op, b)

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.variable_scope(name):
            x_bn = \
                tf.contrib.layers.batch_norm(
                    inputs=x,
                    decay=0.9,
                    center=True,
                    scale=True,
                    epsilon=1e-5,
                    updates_collections=None,
                    is_training=self.mode == 'train',
                    fused=True,
                    data_format='NHWC',
                    zero_debias_moving_mean=True,
                    scope='BatchNorm'
                )

        return x_bn

    def _leaky_relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _max_pool(self, x, ksize, strides):
        return tf.nn.max_pool(x,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, strides, strides, 1],
                              padding='SAME',
                              name='max_pool')
