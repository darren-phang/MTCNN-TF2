# @Author : Darren Pang
# @Time : 2020/8/30 1:55 下午
# @File : trainer.py
import os
import models
import tensorflow as tf
from utils import cal_accuracy
from tqdm import tqdm


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,
                 warmup_steps,
                 warmup_start,
                 initial_learning_rate,
                 decay_steps,
                 alpha):
        super(WarmupCosineDecay, self).__init__()
        self.warmup_steps = warmup_steps
        self.warmup_start = warmup_start
        self.lr_schedule = tf.keras.experimental.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            alpha=alpha
        )
        self.increasing = (initial_learning_rate - warmup_start) / warmup_steps

    @tf.function
    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        if step < self.warmup_steps:
            return tf.cast(self.warmup_start + self.increasing * step, dtype=tf.float32)
        else:
            return self.lr_schedule(step - self.warmup_steps)


class Trainer:
    def __init__(self,
                 model,
                 generator,
                 model_name,
                 logdir,
                 batch_size=384,
                 epoch=30,
                 learning_rate=0.001,
                 step_pre_epoch=2000):
        self.model = model
        self.generator = generator.get_generator(batch_size)
        self.global_epoch = 1
        self.global_step = 1
        self.epoch = epoch
        self.steps_pre_epoch = step_pre_epoch

        lr_schedule = WarmupCosineDecay(warmup_start=learning_rate * 0.001,
                                        warmup_steps=2 * self.steps_pre_epoch,
                                        initial_learning_rate=learning_rate,
                                        decay_steps=(epoch - 2) * self.steps_pre_epoch,
                                        alpha=0.0)
        self.optimizer = tf.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
        model_save_dict = {model_name: self.model}
        self.ckpt = tf.train.Checkpoint(**model_save_dict)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, logdir, max_to_keep=3)
        self.summary_writer = tf.summary.create_file_writer(os.path.join(logdir, "summary"), name="train")
        self.set_describe()

    @tf.function
    def train_step(self, _inp):
        raise NotImplementedError

    def set_describe(self):
        self.describe = "Epoch {}, acc: {}, cls loss: {}, bbox loss: {}"

    def train(self):
        train_process = tqdm(range(self.steps_pre_epoch), desc="Training: ")
        for _inp in self.generator:
            train_info = self.train_step(_inp)
            self.tensorboard(train_info)
            train_process.set_description(self.describe.format(self.global_epoch, *train_info))
            train_process.update()
            self.global_step += 1
            if self.global_step % self.steps_pre_epoch == 0:
                self.ckpt_manager.save(self.global_epoch)
                self.global_epoch += 1
                train_process.close()
                train_process = tqdm(range(self.steps_pre_epoch), desc="Training: ")
            if self.global_epoch > self.epoch:
                train_process.close()
                break

    def tensorboard(self, train_info):
        pass
