# -*- coding: utf-8 -*-
"""Assignent 2_deep_learning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KLJYbxqkZT4I4rufPJLDP5lCCy_pZ6Di
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import to_categorical

!pip install tensorflow==1.14.0

from google.colab import drive
drive.mount('/content/drive')

!ls "/content/drive/My Drive"

df = pd.read_csv('/content/drive/My Drive/Datasets/fer2013.csv')
df.head

df = df[df["Usage"]=="Training"]
df.head

df.pixels = df.pixels.apply(lambda x:x.split(" "))

df.pixels[0]
df.pixels = df.pixels.apply(lambda x: [int(i) for i in x])
df.pixels

X = np.zeros((len(df),48*48)).astype('float32')
X.shape
for i in range(len(df)):
  X[i]=df['pixels'][i]
X.shape
X= X/255.0
X[0].shape

Y = df.emotion
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)

X_train.shape
Y_train = to_categorical(Y_train, num_classes=7)
Y_test = to_categorical(Y_test, num_classes=7)

from tensorflow.keras import Sequential, models
from tensorflow.keras.layers import Dense,Dropout,Conv2D,BatchNormalization,Flatten,MaxPool2D,Activation

X_train = X_train.reshape(-1,48,48,1)
X_test = X_test.reshape(-1,48,48,1)

input_shape=(48,48,1)

X_train.shape

from tensorflow.keras import models,layers

model = models.Sequential()
model.add(layers.Conv2D(64, (5,5),input_shape=input_shape, activation='relu'))
model.add(layers.Conv2D(64, (5,5), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2, 2)))

model.summary()

model.add(layers.Conv2D(128, (5, 5), activation='relu'))
model.add(layers.Conv2D(128, (5, 5), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2, 2)))

model.summary()

model.add(layers.Conv2D(256, (1,1), activation='relu'))
model.add(layers.Conv2D(256, (1,1), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2, 2)))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(128))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(7,activation='softmax'))

model.summary()

epoch=20
batch_size = 64
learning_rate = 0.001

from tensorflow.keras.optimizers import Adam

opt = Adam(learning_rate=learning_rate)
model.compile(optimizer=opt, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(X_train,Y_train, epochs=epoch, batch_size=batch_size, validation_data=(X_test, Y_test))

Y_test = np.argmax(Y_test,axis=1)
Y_test

from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

#lr=ReduceLROnPlateau(patience=3)
#es = EarlyStopping(patience=3)
#im_gen = ImageDataGenerator()

#model_augmented = model.fit_generator(im_gen.flow(X_train,Y_train),
#                                      steps_per_epoch=len(X_train)/64,
 #                                     epochs=epoch,
 #                                     callbacks=[lr,es],
 #                                     validation_data=(X_test, Y_test),
 #                                     )

"""Confusion Matrix on the trained model"""

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)
y_pred

confu_mat = confusion_matrix(Y_test, y_pred)
confu_mat

classes = ['Angry', 'Disgust', 'Fear', "Happy", "Sad", "Surprise", "Neutral"]
confusion_df = pd.DataFrame(confu_mat, columns=classes, index=classes)
confusion_df

"""Save the file to .h5"""

model.save("Emotions.h5")

"""Converting Saved Model to .pb model"""

from keras import backend as K
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

tf.train.write_graph(frozen_graph, 'model', 'Emotion Detector.pb', as_text=False)
