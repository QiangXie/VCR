from model import train
import config
from data import VinAnnotationsReader
import os
import shutil

FLAGS = config.FLAGS

def data_augmentation(data_dir):
    vin_annotations_reader = VinAnnotationsReader.VinAnnotationsReader(data_dir)
    vin_annotations_reader.crop(os.path.join(data_dir, "crop"))
    vin_annotations_reader.data_augmentation(os.path.join(data_dir, "crop"),
            os.path.join(data_dir, "augment"),
            multiple=20)


if __name__ == "__main__":
    data_augmentation("/home/new/Data/vin_data_checked/train")
    data_augmentation("/home/new/Data/vin_data_checked/val")
    dirs = ["/home/new/Data/vin_data_checked/train/augment", "/home/new/Data/vin_data_checked/train/augment"]
    for dir_ in dirs:
        files = os.listdir(dir_)
        for i in files:
            ori = os.path.join(dir_, i)
            dst = os.path.join("/home/new/Data/vin_data_checked/mix", i)
            shutil.copy(ori, dst)


    trainer = train.Trainer(train_data_dir="/home/new/Data/vin_data_checked/mix", 
            val_data_dir="/home/new/Data/vin_data_checked/val/augment", 
            log_dir=FLAGS.log_dir,
            batch_size=FLAGS.batch_size,
            step_set0=FLAGS.step_set0,
            restore=FLAGS.restore,
            checkpoint_dir=FLAGS.checkpoint_dir,
            train_epochs=FLAGS.num_epochs,
            save_step_interval=10000,
            validation_interval_steps=FLAGS.validation_interval_steps,
            image_height=FLAGS.image_height,
            image_width=FLAGS.image_width,
            image_channel=FLAGS.image_channel,
            out_channels=FLAGS.out_channels,
            leakiness=FLAGS.leakiness,
            num_hidden=FLAGS.num_hidden,
            output_keep_prob=FLAGS.output_keep_prob,
            num_classes=FLAGS.char_classes,
            initial_learning_rate=1e-4,
            decay_epoch=30,
            decay_rate=FLAGS.decay_rate,
            beta1=FLAGS.beta1,
            beta2=FLAGS.beta2,
            device="/gpu:0",
            data_type=3)
    trainer.train()
