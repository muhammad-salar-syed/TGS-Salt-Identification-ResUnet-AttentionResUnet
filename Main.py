

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

seed=24
batch_size=16

def Generator(img_path,mask_path):
    
    frame_args=dict(rescale=1./255,
                    horizontal_flip=True,
                    vertical_flip=True)
    mask_args=dict(rescale=1./255,
                   horizontal_flip=True,
                   vertical_flip=True)
    frame_datagen = ImageDataGenerator(**frame_args)
    mask_datagen = ImageDataGenerator(**mask_args)
    
    frame_generator = frame_datagen.flow_from_directory(
        img_path,
        class_mode = None,
        color_mode = 'rgb',
        target_size = (96,128),
        batch_size = batch_size,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        mask_path,
        class_mode = None,
        color_mode = 'grayscale',
        target_size = (96,128),
        batch_size = batch_size,
        seed=seed)
    
    train_generator=zip(frame_generator,mask_generator)
    
    for (frame,mask) in train_generator:
        yield (frame,mask)

train_img_path ='./Data/Train/Images/'
train_mask_path='./Data/Train/Masks/'

val_img_path='./Data/Test/Images/'
val_mask_path='./Data/Test/Masks/'

train_img_gen = Generator(train_img_path,train_mask_path)
val_img_gen=Generator(val_img_path,val_mask_path)

steps_per_epoch = len(os.listdir('./Data/Train/Images/train/'))//batch_size
val_steps_per_epoch = len(os.listdir('./Data/Test/Images/test/'))//batch_size

import segmentation_models as sm
from keras.callbacks import EarlyStopping
from models import Unet,ResUnet,Attention_ResUNet

early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
loss = sm.losses.BinaryFocalLoss()
metrics = [sm.metrics.FScore(threshold=0.5)]

input_shape=(96,128,3)
model=Attention_ResUNet(input_shape)
model.compile(optimizer='adam', loss=loss, metrics=metrics)
model.summary()

history=model.fit(train_img_gen,
          steps_per_epoch=steps_per_epoch,
          epochs=50,
          verbose=1,
          validation_data=val_img_gen,
          validation_steps=val_steps_per_epoch,
          callbacks=early_stop)

model.save('./AttentionResunet.hdf5')
