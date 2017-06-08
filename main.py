'''TensorFlow implementation of http://arxiv.org/pdf/1312.6114v10.pdf'''

from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
from tensorflow.examples.tutorials.mnist import input_data

from progressbar import ETA, Bar, Percentage, ProgressBar

from vae import VAE
from gan import GAN
from dataset import load_dataset

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_string("dataset", "MNIST", "subdirectory of working_directory containing image data")
flags.DEFINE_string("img_extension", "*", "Only load files from the dataset directory " +
        "having this extension (e.g. 'png', 'jpg'). '*' tries to load all files.")
flags.DEFINE_integer("channels", 0, "How many colour channels to read. 1: grayscale, "+
        "3: RGB, 4: RGBA, 0: infer automatically")
flags.DEFINE_integer("hidden_size", 128, "size of the hidden VAE unit")
flags.DEFINE_string("model", "gan", "gan or vae")

FLAGS = flags.FLAGS

if __name__ == "__main__":
    data_directory = os.path.join(FLAGS.working_directory, FLAGS.dataset)
    if FLAGS.dataset == 'MNIST':
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = input_data.read_data_sets(data_directory, one_hot=True).train
    else:
        dataset = load_dataset(FLAGS.dataset, FLAGS.img_extension, FLAGS.channels)

    img = dataset.sample_img

    assert FLAGS.model in ['vae', 'gan']
    if FLAGS.model == 'vae':
        model = VAE(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate)
    elif FLAGS.model == 'gan':
        model = GAN(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate)

    for epoch in range(FLAGS.max_epoch):
        training_loss = 0.0

        pbar = ProgressBar()
        for i in pbar(range(FLAGS.updates_per_epoch)):
            images, _ = mnist.train.next_batch(FLAGS.batch_size)
            loss_value = model.update_params(images)
            training_loss += loss_value

        training_loss = training_loss / \
            (FLAGS.updates_per_epoch * FLAGS.batch_size)

        print("Loss %f" % training_loss)

        model.generate_and_save_images(
            FLAGS.batch_size, FLAGS.working_directory)
