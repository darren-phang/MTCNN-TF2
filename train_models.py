import models
from dataloader import MTCNNGenerator
import tensorflow as tf
from utils import cal_accuracy
from tqdm import tqdm
import os


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
                 learning_rate=0.001):
        self.model = model
        self.generator = generator.get_generator(batch_size)
        self.global_epoch = 1
        self.global_step = 1
        self.epoch = epoch
        self.steps_pre_epoch = generator.image_number // batch_size

        lr_schedule = WarmupCosineDecay(warmup_start=learning_rate * 0.001,
                                        warmup_steps=5 * self.steps_pre_epoch,
                                        initial_learning_rate=learning_rate,
                                        decay_steps=(epoch - 5) * self.steps_pre_epoch,
                                        alpha=0.0)
        self.optimizer = tf.optimizers.SGD(learning_rate=lr_schedule)
        model_save_dict = {model_name: self.model}
        self.ckpt = tf.train.Checkpoint(**model_save_dict)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, logdir, max_to_keep=3)
        self.summary_writer = tf.summary.create_file_writer(os.path.join(logdir, "summary"), name="train")

    @tf.function
    def train_step(self, _inp):
        with tf.GradientTape() as tape:
            imgs, bboxes, labels = _inp
            cls_prob, bbox_pred = self.model(imgs, True)
            cls_loss = models.cls_ohem(cls_prob, labels)
            bbox_loss = models.bbox_ohem(bbox_pred, bboxes, labels)
            accuracy = cal_accuracy(cls_prob, labels)
            regular_loss = tf.add_n(self.model.losses)
            loss = cls_loss + bbox_loss * 0.5 + regular_loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))
        return cls_loss, bbox_loss, accuracy

    def train(self):
        train_process = tqdm(range(self.steps_pre_epoch), desc="Training: ")
        describe = "Epoch {}, acc: {}, cls loss: {}, bbox loss: {}"
        for _inp in self.generator:
            cls_loss, bbox_loss, accuracy = self.train_step(_inp)
            self.tensorboard(cls_loss, bbox_loss, accuracy)
            train_process.set_description(describe.format(self.global_epoch, accuracy, cls_loss, bbox_loss))
            train_process.update()
            self.global_step += 1
            if self.global_step % self.steps_pre_epoch == 0:
                self.ckpt_manager.save(self.global_epoch)
                self.global_epoch += 1
            if self.global_epoch > self.epoch:
                break
        train_process.close()

    def tensorboard(self, cls_loss, bbox_loss, accuracy):
        with self.summary_writer.as_default():
            current_lr = self.optimizer._get_hyper('learning_rate')(self.optimizer._iterations)
            tf.summary.scalar("recode/learning_rate", current_lr, step=self.global_epoch)
            tf.summary.scalar("loss/cls_loss", cls_loss, step=self.global_epoch)
            tf.summary.scalar("loss/bbox_loss", bbox_loss, step=self.global_epoch)
            tf.summary.scalar("recode/accuracy", accuracy, step=self.global_epoch)


dataset_path = '/media/cdut9c403/新加卷/darren/wider_face'
generator = MTCNNGenerator(dataset_path, "PNet")

pnet = models.PNet()
trainer = Trainer(pnet, generator, "pnet", "/media/cdut9c403/新加卷/darren/logs/MTCNN/Pnet", 384, 30, 0.001)
trainer.train()
