'''
Unet basic implementation
'''

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K


def Unet(input_size = (256,256,1), k_size=3):
    merge_axis = -1 # Feature maps are concatenated along last axis (for tf backend)
    data = Input(shape=input_size)
    conv1 = Convolution2D(filters=32, kernel_size=k_size, padding='same', activation='relu')(data)
    conv1 = Convolution2D(filters=32, kernel_size=k_size, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(pool1)
    conv2 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(pool2)
    conv3 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(pool3)
    conv4 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
	
    conv5 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(pool4)
    conv5 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = Convolution2D(filters=512, kernel_size=k_size, padding='same', activation='relu')(pool5)
    conv6 = Convolution2D(filters=512, kernel_size=k_size, padding='same', activation='relu')(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)
	
    conv7 = Convolution2D(filters=1024, kernel_size=k_size, padding='same', activation='relu')(pool6)

    up1 = UpSampling2D(size=(2, 2))(conv7)
    conv8 = Convolution2D(filters=1024, kernel_size=k_size, padding='same', activation='relu')(up1)
    conv8 = Convolution2D(filters=1024, kernel_size=k_size, padding='same', activation='relu')(conv8)
    merged1 = concatenate([conv6, conv8], axis=merge_axis)
    conv8 = Convolution2D(filters=512, kernel_size=k_size, padding='same', activation='relu')(merged1)

    up2 = UpSampling2D(size=(2, 2))(conv8)
    conv9 = Convolution2D(filters=1024, kernel_size=k_size, padding='same', activation='relu')(up2)
    conv9 = Convolution2D(filters=1024, kernel_size=k_size, padding='same', activation='relu')(conv9)
    merged2 = concatenate([conv5, conv9], axis=merge_axis)
    conv9 = Convolution2D(filters=1024, kernel_size=k_size, padding='same', activation='relu')(merged2)

    up3 = UpSampling2D(size=(2, 2))(conv9)
    conv10 = Convolution2D(filters=512, kernel_size=k_size, padding='same', activation='relu')(up3)
    conv10 = Convolution2D(filters=512, kernel_size=k_size, padding='same', activation='relu')(conv10)
    merged3 = concatenate([conv4, conv10], axis=merge_axis)
    conv10 = Convolution2D(filters=512, kernel_size=k_size, padding='same', activation='relu')(merged3)

    up4 = UpSampling2D(size=(2, 2))(conv10)
    conv11 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(up4)
    conv11 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(conv11)
    merged4 = concatenate([conv3, conv11], axis=merge_axis)
    conv11 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(merged4)
	
    up5 = UpSampling2D(size=(2, 2))(conv11)
    conv12 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(up5)
    conv12 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(conv12)
    merged5 = concatenate([conv2, conv12], axis=merge_axis)
    conv12 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(merged5)
		
    up6 = UpSampling2D(size=(2, 2))(conv12)
    conv13 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(up6)
    conv13 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv13)
    merged6 = concatenate([conv1, conv13], axis=merge_axis)
    conv13 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(merged6)
	
    conv14 = Convolution2D(filters=1, kernel_size=k_size, padding='same', activation='sigmoid')(conv13)

    output = conv14
    
    return Model(inputs=data, outputs=output)
  

if __name__=="__main__":
    model = Unet(input_size=(256,256,1))

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #model.summary()
    pretrained_weights=None
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    print(model.summary())