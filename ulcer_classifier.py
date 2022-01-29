
import matplotlib
from matplotlib import pylab as plt
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
import scipy.io as sio
import numpy as np

from imgaug import augmenters as iaa

import os
train_loss=[]
val_loss=[]
train_acc=[]
val_acc=[]
dpData = 'bm586_data' # training data location
dpRoot = 'BM586'
dpTest = 'test_data' # test data location
batch_size = 32
image_size=(580,580)

#%%
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x=inputs

    x = layers.Conv2D(8, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)


    for size in [16,32,64]:
       # x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(size, 3, padding="same",
        kernel_regularizer=keras.regularizers.L2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        x = layers.Conv2D(size, 3, padding="same",kernel_regularizer=keras.regularizers.L2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
      
        
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=keras.regularizers.L2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    
    units = num_classes

    x = layers.Dropout(0.5)(x)
                       
    outputs = layers.Dense(units, activation="softmax")(x)
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,), num_classes=4)
keras.utils.plot_model(model, show_shapes=True) # constructing the model

#%% loading the data
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
        fill_mode='nearest')

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.15)          

test_datagen = ImageDataGenerator(rescale=1./255)

train_ds = train_datagen.flow_from_directory(
    dpData,
    subset="training",
    shuffle=True,
    seed=1337,
    batch_size=batch_size,
)

val_ds = val_datagen.flow_from_directory(
    dpData,
    subset="validation",
    shuffle=True,
    seed=1337,
    batch_size=batch_size,
)

test_ds = test_datagen.flow_from_directory(
    dpTest,    
    batch_size=batch_size,
   
)
#%% compiling the model
#model.load_weights(dpRoot+"/last_sgd_tumdata_l2reg.h5")
adam_opt = Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd_opt=SGD(lr=0.0001)
model.compile(
    optimizer=adam_opt,
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

   #%% training process
checkpoint = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
history=model.fit_generator(
    generator=train_ds, epochs=50, callbacks=[checkpoint],validation_data=val_ds,
)

#%%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
#%%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
#%%
model.save_weights(dpRoot+"/classifier_weights.h5")


