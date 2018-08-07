from model import train
import config

FLAGS = config.FLAGS

if __name__ == "__main__":
    trainer = train.Trainer(train_data_dir=FLAGS.train_data_dir, 
            val_data_dir=FLAGS.val_data_dir, 
            log_dir=FLAGS.log_dir,
            batch_size=FLAGS.batch_size,
            step_set0=FLAGS.step_set0,
            restore=FLAGS.restore,
            checkpoint_dir=FLAGS.checkpoint_dir,
            train_epochs=FLAGS.num_epochs,
            save_step_interval=FLAGS.save_step_interval,
            validation_interval_steps=FLAGS.validation_interval_steps,
            image_height=FLAGS.image_height,
            image_width=FLAGS.image_width,
            image_channel=FLAGS.image_channel,
            out_channels=FLAGS.out_channels,
            leakiness=FLAGS.leakiness,
            num_hidden=FLAGS.num_hidden,
            output_keep_prob=FLAGS.output_keep_prob,
            num_classes=FLAGS.char_classes,
            initial_learning_rate=FLAGS.initial_learning_rate,
            decay_epoch=FLAGS.decay_epoch,
            decay_rate=FLAGS.decay_rate,
            beta1=FLAGS.beta1,
            beta2=FLAGS.beta2,
            device="/gpu:0")
    trainer.train()
