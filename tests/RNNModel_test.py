#!/usr/bin/env python
# coding: utf-8
# Author : Rahul Bhadani
# Initial Date: 2023-05-17
__author__ = 'Rahul Bhadani'
__email__  = 'rahulbhadani@email.arizona.edu'


import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Get the data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# Set TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

Data = pd.read_csv("/home/refulgent/VersionControl/Jmscslgroup/PandaData/2020_03_03/2020-03-03-15-36-24-479038_states.csv")

Data = Data.drop("Clock", axis=1)
Data['Time'] = Data['Time'] - Data['Time'].iloc[0]
X = Data.drop("acc_status", axis = 1)
Y = Data[["acc_status"]]

scaler = MinMaxScaler()
input_features = scaler.fit_transform(X)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(Y)

                                      
# Determine the index to split the data
split_index = int(len(X) * 0.8)  # 80% for training, 20% for testing

# Split the data and labels
X_train = input_features[:split_index]
X_test = input_features[split_index:]
y_train = encoded_labels[:split_index]
y_test = encoded_labels[split_index:]

from LESim import RNNModel

rnn = RNNModel(hidden_units=256, input_size=X_train.shape[1])

rnn.compile_model(optimizer='adam', loss_fn='categorical_crossentropy')


X_train_reshaped = np.expand_dims(X_train, axis=1)
X_test_reshaped = np.expand_dims(X_test, axis=1)

rnn.train_model(input_X = X_train_reshaped, labels_y=y_train, epochs=15, batch_size=32)
predicted_y = rnn.predict(X_test_reshaped)

decoded_predictions = label_encoder.inverse_transform(np.argmax(predicted_y, axis=1))