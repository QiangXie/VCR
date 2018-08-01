import tensorflow as tf 

tf.app.flags.DEFINE_integer('out_channels', 64, 'output channels of last layer in CNN')
tf.app.flags.DEFINE_integer('image_channel', 3, 'image channels as input')
tf.app.flags.DEFINE_integer('image_width', 384, 'image width')
tf.app.flags.DEFINE_integer('image_height', 64, 'image height')
tf.app.flags.DEFINE_integer('char_classes', 35, 'char classes')
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size')
tf.app.flags.DEFINE_integer('num_hidden', 128, 'lstm num hidden')

tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, 'inital lr')
tf.app.flags.DEFINE_float('decay_rate', 0.1, 'the lr decay rate')
tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')
tf.app.flags.DEFINE_integer('decay_epoch', 50, 'the lr decay_step for optimizer')
tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')
tf.app.flags.DEFINE_integer('cnn_count', 4, 'count of cnn module to extract image features.')
tf.app.flags.DEFINE_float('leakiness', 0.01, 'leakiness of lrelu')
tf.app.flags.DEFINE_float('output_keep_prob', 0.8, 'output_keep_prob in lstm')

tf.app.flags.DEFINE_string('log_dir', "./log", "path to save log file")
tf.app.flags.DEFINE_string('train_data_dir', "/home/new/Data/vin_data_checked/train", "train data path")
tf.app.flags.DEFINE_string('val_data_dir', "/home/new/Data/vin_data_checked/val", "val data path")
tf.app.flags.DEFINE_bool('restore', True, "whether to restore graph from checkpoint")
tf.app.flags.DEFINE_string('checkpoint_dir', "./ckpt", "path to save checkpoint")
tf.app.flags.DEFINE_integer('save_step_interval', 8000, 'save checkpoint step interval')
tf.app.flags.DEFINE_integer('validation_interval_steps', 4000, 'the steps interval to validation')
tf.app.flags.DEFINE_integer('num_epochs', 150, 'num of epochs to train')
tf.app.flags.DEFINE_integer('augment_factor', 20, 'num of epochs to train')
tf.app.flags.DEFINE_bool('step_set0', True, "whether to restore graph from checkpoint")


FLAGS = tf.app.flags.FLAGS
