#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################
#
# File Name:  estimator_custom.py
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
# Create Time:    2017-08-18 15:04:33
#
######################################################

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os
import time
from datetime import datetime, timedelta
import numpy as np
import tensorflow as tf

def model_fn(features, labels, mode):

    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)

    y = W * features["x"] + b

    loss = tf.reduce_sum(tf.square(y - labels))

    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)

    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    return tf.estimator.EstimatorSpec(mode=mode, predictions=y, loss=loss, train_op=train)


estimator = tf.estimator.Estimator(model_fn = model_fn)

x_train = np.array([1.0, 2.0, 3.0, 4.0])
y_train = np.array([0.0, -1.0, -2.0, -3.0])

x_eval = np.array([2.0, 5.0, 8.0, 1.0])
y_eval = np.array([-1.01, -4.1, -7.0, 0.0])

input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

estimator.train(input_fn=input_fn, steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("train metrics: %r" % train_metrics)
print("eval metrics: %r" % eval_metrics)



