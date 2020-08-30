# @Author : Darren Pang
# @Time : 2020/8/30 1:53 下午
# @File : train_models_landmark.py
import models
import tensorflow as tf
from train.trainer import Trainer
from dataloader import LandmarkGenerator


class LandmarkTrain(Trainer):
    def set_describe(self):
        self.describe = "Epoch {}, landmark loss: {}"

    @tf.function
    def train_step(self, _inp):
        with tf.GradientTape() as tape:
            imgs, landmark_target = _inp
            _, _, landmark_pred = self.model(imgs, True)
            landmark_loss = models.landmark_ohem(landmark_pred, landmark_target)
            regular_loss = tf.add_n(self.model.losses)
            loss = landmark_loss + regular_loss
        grads = tape.gradient(loss, self.model.landmark.trainable_variables)
        self.optimizer.apply_gradients(list(zip(grads, self.model.landmark.trainable_variables)))
        return (landmark_loss, )

    def tensorboard(self, train_info):
        with self.summary_writer.as_default():
            current_lr = self.optimizer._get_hyper('learning_rate')(self.optimizer._iterations)
            tf.summary.scalar("recode/learning_rate", current_lr, step=self.global_step)
            tf.summary.scalar("loss/landmark", train_info[0], step=self.global_step)


dataset_path = '/Users/darrenpang/Documents/datasets/LFW/train'
log_path = '../ckpt/my/Onet-landmark'
onet_ckpt = '../ckpt/my/Onet/ckpt-18'
model_name = 'onet'
generator = LandmarkGenerator(dataset_path, 'landmark')
model = models.ONet()

trainer = LandmarkTrain(model, generator, model_name, log_path,
                        500, 18, 0.005, step_pre_epoch=2000)
trainer.ckpt.restore(onet_ckpt)
trainer.train()
