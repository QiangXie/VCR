from model import train
import config

FLAGS = config.FLAGS



if __name__ == "__main__":
    trainer = train.Trainer(FLAGS.train_data_dir, FLAGS.val_data_dir, "/gpu:0")
    trainer.train()
