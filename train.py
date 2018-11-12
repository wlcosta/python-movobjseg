# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 16:24:19 2018

@author: rrb
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.layers import Input, Dense, Conv2D, Activation, MaxPooling2D
from keras.models import Model
import keras.backend as K

BASE_DIR = 'E:\wlc2\MovingObjectSegmentation-master\MovingObjectSegmentation-master\CDNet\CDNetDataset'

video = 'baseline\\highway'
method = 'manual'
frames = 200

def training(video, method, frames):
    
    BATCH_SIZE = 5
    NUMEPOCHS = 20
    LR = 1e-3
    HALF_SIZE = 15
    
    imgDir = os.path.join(BASE_DIR, method, str(frames)+ 'frames', video, 'input')
    labelDir = os.path.join(BASE_DIR, method, str(frames)+ 'frames', video, 'GT')
    
    set, names, labels = getImdb(imgDir, labelDir)
    
    mask = plt.imread(os.path.join(
            BASE_DIR, method, str(frames)+ 'frames', video, 
            'ROI.bmp')
    ).copy()
    
    mask = mask[:, :, 0]
    A = np.max(mask)
    
    mask[mask == A] = 1
    if(mask.shape[0] > 400) or (mask.shape[1] > 400):
        mask = cv2.resize(
                mask, dsize=(
                int(mask.shape[0]*0.5),
                int(mask.shape[1]*0.5)),
                interpolation=cv2.INTER_NEAREST)
    
    inputs = Input(shape=(320, 240, 3))
    x = Conv2D(32, (7, 7), strides=1, padding='valid')(inputs)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=1)(x)
    x = Conv2D(32, (7, 7), strides=1, padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (7, 7), strides=1, padding='valid')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=1)(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (7, 7), strides=1, padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (7, 7), strides=1, padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (7, 7), strides=1, padding='valid')(x)
    output = Dense(1, activation='sigmoid')(x)
    
    def euclidean_distance_loss(y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='rmsprop', loss=euclidean_distance_loss, metrics=['accuracy'])
    
    
    
def getImdb(imgDir, labelDir):
    files = []
    for file in os.listdir(imgDir):
        if file.endswith(".jpg"):
            files.append(file)
    
    labelFiles = []
    for file in os.listdir(labelDir):
        if file.endswith(".png"):
            labelFiles.append(file)
    
    imset = np.ones(1, len(files))
    
    return imset, files, labelFiles
            
        
    
    
