import os
import time
import random
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
import tensorflow_io as tfio
import tensorflow_model_optimization as tfmot

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import matplotlib.pyplot as plt

import cv2

from collections import OrderedDict
from models.hrnet import HRNetBody, hrnet_body

# @title
'''
    implement Light CNN
    @author: Alfred Xiang Wu
    @date: 2017.07.04
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(
                in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels,
                        kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels,
                         kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels,
                         kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out


class network_29layers_v2(nn.Module):
    def __init__(self, block, layers, num_classes=80013):
        super(network_29layers_v2, self).__init__()
        self.conv1 = mfm(1, 48, 5, 1, 2)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.fc = nn.Linear(8*8*128, 256)
        self.fc2 = nn.Linear(256, num_classes, bias=False)

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block1(x)
        x = self.group1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block2(x)
        x = self.group2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        x = F.dropout(fc, training=self.training)
        out = self.fc2(x)
        return out, fc


def LightCNN_29Layers_v2(**kwargs):
    model = network_29layers_v2(resblock, [1, 2, 3, 4], **kwargs)
    return model


data_dir = "/n/home01/dzhou/data/experiments/100-faces/"
img_size = (61, 61)
batch_size = 64

# yields float32 tensors of shape (batch_size, image_size[0], image_size[1], num_channels)
train_ds = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0,
    seed=6869,
    label_mode=None,
    shuffle=True,
    #color_mode = "grayscale",
    image_size=img_size,
    batch_size=batch_size)


def hrnet_stem(filters=64):
    stem_layers = [layers.Conv2D(filters, 3, 2, 'same'),
                   layers.BatchNormalization(),
                   layers.Conv2D(filters, 3, 2, 'same'),
                   layers.BatchNormalization(),
                   layers.Activation('relu')]

    def forward(x):
        for layer in stem_layers:
            x = layer(x)
        return x

    return forward


def hrnet_heads(input_channels=64, output_channels=17):
    # Construct up sacling layers.
    scales = [2, 4, 8]
    up_scale_layers = [layers.UpSampling2D((s, s)) for s in scales]
    concatenate_layer = layers.Concatenate(axis=3)
    heads_layers = [layers.Conv2D(filters=input_channels, kernel_size=(1, 1),
                                  strides=(1, 1), padding='same'),
                    layers.BatchNormalization(),
                    layers.Activation('relu'),
                    layers.Conv2D(filters=output_channels, kernel_size=(1, 1),
                                  strides=(1, 1), padding='same')]

    def forward(inputs):
        scaled = [f(x) for f, x in zip(up_scale_layers, inputs[1:])]
        x = concatenate_layer([inputs[0], scaled[0], scaled[1], scaled[2]])
        for layer in heads_layers:
            x = layer(x)
        return x

    return forward


def hrnet_v2(input_shape, output_channels, width=18, name="hrnetv2"):
    """This function returns a functional model of HRNetV2.
    Args:
        width: the hyperparameter width.
        output_channels: number of output channels.
    Returns:
        a functional model.
    """
    # Get the output size of the HRNet body.
    last_stage_width = sum([width * pow(2, n) for n in range(4)])

    # Describe the model.
    inputs = keras.Input(input_shape, dtype=tf.float32)
    x = hrnet_stem(64)(inputs)
    x = hrnet_body(width)(x)
    outputs = hrnet_heads(input_channels=last_stage_width,
                          output_channels=output_channels)(x)

    # Construct the model and return it.
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)

    model.compile(optimizer=keras.optimizers.Adam(0.001, amsgrad=True, epsilon=0.001),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=[keras.metrics.MeanSquaredError()])

    return model


class YourFaceModel(tf.keras.Model):

    def __init__(self, img_size):
        super(YourFaceModel, self).__init__()

        lc1 = layers.LocallyConnected2D(
            32, (7, 7), strides=(1, 1), activation=tf.nn.relu)
        mp1 = layers.MaxPool2D(pool_size=(2, 2), strides=None)
        fl1 = layers.Reshape((1, 1, -1))
        fc1 = layers.Dense(3600, activation=tf.nn.relu)
        fl2 = layers.Reshape((60, 60, 1))
        lc2 = layers.LocallyConnected2D(
            32, (5, 5), strides=(1, 1), activation=tf.nn.relu)
        mp2 = layers.MaxPool2D(pool_size=(3, 3), strides=None)
        fl3 = layers.Reshape((1, 1, -1))
        fc2 = layers.Dense(3721, activation=tf.nn.relu)
        fl4 = layers.Reshape((61, 61, 1))

        # task 2
        fl45 = layers.Reshape((1, 1, -1))
        fc3 = layers.Dense(3600, activation=tf.nn.relu)
        fl5 = layers.Reshape((60, 60, 1))
        lc3 = layers.LocallyConnected2D(
            32, (5, 5), strides=(1, 1), activation=tf.nn.relu)
        mp3 = layers.MaxPool2D(pool_size=(3, 3), strides=None)
        fl6 = layers.Reshape((1, 1, -1))
        fc4 = layers.Dense(3721, activation=tf.nn.relu)
        fl7 = layers.Reshape((61, 61, 1))

        task_layers = [lc1, mp1, fl1, fc1, fl2, lc2, mp2, fl3, fc2, fl4]
        self.first_task = keras.Sequential(task_layers)

        auxiliary_layers = [fl45, fc3, fl5, lc3, mp3, fl6, fc4, fl7]
        self.aux_task = keras.Sequential(auxiliary_layers)

        self.first_task.build((batch_size, img_size[0], img_size[1], 3))
        self.aux_task.build((batch_size, img_size[0], img_size[1], 1))

        self.keypoint_model = hrnet_v2(input_shape=(img_size[0], img_size[1], 1),
                                       output_channels=98, width=18, name="hrnetv2")
        self.keypoint_model.build((batch_size, img_size[0], img_size[1], 1))

    def call(self, inputs):
        x = self.first_task(inputs)
        x2 = self.aux_task(x)
        return x, x2, self.keypoint_model(x2)


def light_CNN():
    model = LightCNN_29Layers_v2()
    checkpoint = torch.load(
        '/n/home01/dzhou/data/experiments/data/LightCNN_29Layers_V2_checkpoint.pth.tar')

    state_dict = OrderedDict()

    for (key, value) in checkpoint['state_dict'].items():
        new_key = '.'.join(key.split('.')[1:])
        state_dict[new_key] = value

    model.load_state_dict(state_dict)
    return model


def kp_model(model_dir):
    model = tf.keras.models.load_model(model_dir)
    return model


def get_facial_recog_features(input_img, cnn_model):
    transform = transforms.Compose([transforms.ToTensor()])
    input = torch.zeros(1, 1, 128, 128)
    img = input_img.numpy()

    if img.shape[-1] != 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (128, 128))
    img = np.reshape(img, (128, 128, 1))
    img = transform(img)
    input[0, :, :, :] = img

    input_var = torch.autograd.Variable(input, volatile=True)
    _, features = cnn_model(input_var)
    return features


def keypoint_loss(input, output, kp_model):
    ground_truth = kp_model(input)
    return tf.math.reduce_sum(tf.math.squared_difference(output, ground_truth))


def aux_task_loss(input, output):
    return tf.math.reduce_sum(tf.math.squared_difference(output, input))


def loss_fn(input, output, cnn_model):
    # make our own loss function considering
    # run output thru head pose estimator and check difference from facing on
    # feature distances between facial recognition on original image vs generated image

    #input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    loss = 0

    output_lp = tfio.experimental.filter.laplacian(output, ksize=3)
    lapl_loss = tf.math.reduce_sum(tf.math.squared_difference(
        output_lp, tf.reverse(output_lp, [2])))

    sym_loss = tf.math.reduce_sum(
        tf.math.squared_difference(output, tf.reverse(output, [2])))

    feature_loss = 0
    for i in range(output.shape[0]):
        output_features = get_facial_recog_features(
            output[i, :, :, :], cnn_model).detach().numpy()
        input_features = get_facial_recog_features(
            input[i, :, :, :], cnn_model).detach().numpy()
        feature_loss += tf.math.reduce_sum(tf.math.squared_difference(tf.convert_to_tensor(output_features),
                                                                      tf.convert_to_tensor(input_features)))

    loss = sym_loss + lapl_loss
    # N times height times width times channels
    loss /= tf.math.reduce_prod(tf.cast(tf.shape(output), tf.float32))
    loss += feature_loss / \
        tf.cast(tf.math.reduce_prod(tf.convert_to_tensor(
            output_features.shape)), tf.float32)

    return loss

# =======================PARAMETERS==============================================


lr = 0.01
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=lr,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

num_epochs = 80
gamma1 = 0.33
gamma2 = 0.33
gamma3 = 0.33

checkpoint_path = "cp-yourface-keypoints-100/"

kp_model = kp_model("/n/home01/dzhou/data/experiments/facial-landmark-detection-hrnet/exported/wflw")

def train(train_ds, num_epochs):
    cnn_model = light_CNN()
    model = YourFaceModel(img_size)
    model.build((batch_size, img_size[0], img_size[0], 3))
    print(model.summary())
    min_loss = float('inf')
    
    loss_value_1 = 0
    loss_value_2 = 0
    loss_value_3 = 0
    loss = 0

    for i in range(num_epochs):
        print("Training epoch", i)
        for batch in tqdm(train_ds):
            batch = batch / 255.0
            with tf.GradientTape() as tape:
                # Forward pass.
                rotated_output, reconstructed_output, keypoint_output = model(
                    batch)

                # Loss value for this batch.
                loss_value_1 = loss_fn(batch, rotated_output, cnn_model)
                loss_value_2 = aux_task_loss(batch, reconstructed_output)
                loss_value_3 = keypoint_loss(batch, keypoint_output, kp_model)
                loss = gamma1*loss_value_1 + gamma2*loss_value_2 + gamma3*loss_value_3

            # Get gradients of loss wrt the weights.
            gradients = tape.gradient(loss, model.trainable_weights)

            # Update the weights of the model.
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

            if loss < min_loss:
                model.save(checkpoint_path + "best_model")
                model.first_task.save(checkpoint_path+"best_first_task")
                min_loss = loss

        if i % 5 == 0:
            model.save(checkpoint_path+"epoch_"+str(i))
            model.first_task.save(checkpoint_path+"first_task_epoch_"+str(i))

        print("Epoch {}, task 1 loss={}, task 2 loss={}, loss={}".format(
            i, loss_value_1.numpy(), loss_value_2.numpy(), loss.numpy()))


train(train_ds, num_epochs)

