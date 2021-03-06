#!/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################
#
# File Name:  boston.py
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
# Create Time:    2017-08-22 16:26:36
#
######################################################

from __future__ import division

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os
import time
from datetime import datetime, timedelta


import itertools

import pandas as pd
import tensorflow as tf

def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
            x = pd.DataFrame({k: data_set[k].values for k in FEATURES}),
            y = pd.Series(data_set[LABEL].values),
            num_epochs = num_epochs,
            shuffle = shuffle
            )

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
                   "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
                    "age", "dis", "tax", "ptratio"]
LABEL = "medv"

training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
                                   skiprows=1, names=COLUMNS)
test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
                                     skiprows=1, names=COLUMNS)

feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

regressor = tf.estimator.DNNRegressor(
        feature_columns = feature_cols,
        hidden_units = [10, 10],
        model_dir = "./boston_model"
        )

regressor.train(input_fn = get_input_fn(training_set), steps = 20000)

ev = regressor.evaluate(input_fn = get_input_fn(test_set, num_epochs = 1, shuffle = False))

print ev


