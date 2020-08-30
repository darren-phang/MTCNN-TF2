import models
import tensorflow as tf
from train.trainer import Trainer
from dataloader import BboxGenerator
from utils import cal_accuracy


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class BBoxTrain(Trainer):
    @tf.function
    def train_step(self, _inp):
        with tf.GradientTape() as tape:
            imgs, bboxes, labels = _inp
            cls_prob, bbox_pred, landmark_pred = self.model(imgs, True)
            cls_loss = models.cls_ohem(cls_prob, labels)
            bbox_loss = models.bbox_ohem(bbox_pred, bboxes, labels)
            accuracy = cal_accuracy(cls_prob, labels)
            regular_loss = tf.add_n(self.model.losses)
            loss = cls_loss + bbox_loss + regular_loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(list(zip(grads, self.model.trainable_variables)))
        return cls_loss, bbox_loss, accuracy

    def tensorboard(self, train_info):
        with self.summary_writer.as_default():
            current_lr = self.optimizer._get_hyper('learning_rate')(self.optimizer._iterations)
            tf.summary.scalar("recode/learning_rate", current_lr, step=self.global_step)
            tf.summary.scalar("loss/cls_loss", train_info[0], step=self.global_step)
            tf.summary.scalar("loss/bbox_loss", train_info[1], step=self.global_step)
            tf.summary.scalar("recode/accuracy", train_info[2], step=self.global_step)


dataset_path = ''
log_path = ''
model_name = 'onet'
generator = BboxGenerator(dataset_path, 'train')
model = models.ONet()  # 每次训练需要更换数据集和模型

trainer = BBoxTrain(model, generator, model_name, log_path,
                    500, 18, 0.005, step_pre_epoch=2000)
trainer.train()
