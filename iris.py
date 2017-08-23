#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################
#
# File Name:  iris.py
#
# Function:   
#
# Usage:  
#
# Input:  
#
# Output:	
#
# Author: wenhai.pan
#
# Create Time:    2017-08-22 13:44:20
#
######################################################

from __future__ import absolute_import
from __future__ import division

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os
import time
from datetime import datetime, timedelta

import tensorflow as tf
import numpy as np


IRIS_TRAINING = "./iris_training.csv"
IRIS_TEST = "./iris_test.csv"


tf.logging.set_verbosity(tf.logging.INFO)

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename = IRIS_TRAINING,
        target_dtype = np.int,
        features_dtype = np.float32
        )

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename = IRIS_TEST,
        target_dtype = np.int,
        features_dtype = np.float32
        )

feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

classifier = tf.estimator.DNNClassifier(
        feature_columns = feature_columns,
        hidden_units = [10, 20, 10],
        n_classes = 3,
        model_dir="./iris_model"
        )

train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": np.array(training_set.data)},
        y = np.array(training_set.target),
        num_epochs = None,
        shuffle=True
        )

classifier.train(input_fn = train_input_fn, steps=2000)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": np.array(test_set.data)},
        y = np.array(test_set.target),
        num_epochs = 1,
        shuffle = False
        )

accuracy_score = classifier.evaluate(input_fn = test_input_fn)

print "\nTest Accuracy:\n"
print accuracy_score

new_samples = np.array(
        [[6.4, 3.2, 4.5, 1.5],
        [5.8, 3.1, 5.0, 1.7]],
        dtype=np.float32)

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": new_samples},
        num_epochs = 1,
        shuffle = False
        )

predictions = list(classifier.predict(input_fn = predict_input_fn))

for item in predictions:
    print item


