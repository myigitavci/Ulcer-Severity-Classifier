

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


import os

dpWeights = "C:/Users/yigitavci/Desktop/BOUN DERS/4.sınıf dönem1/BM586" # weights location
dpTest = 'C:/Users/yigitavci/Desktop/BOUN DERS/4.sınıf dönem1/BM586/test_data' # test data location


batch_size = 16
image_size=(580,580)
#%%
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    #x = data_augmentation(inputs)
    x=inputs
    # Entry block
    x = layers.Conv2D(8, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)


    for size in [16,32,64]:
       # x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        x = layers.Conv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
      
        
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    
    units = num_classes

    x = layers.Dropout(0.5)(x)
                       
    outputs = layers.Dense(units, activation="softmax")(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=4)

model.load_weights(dpWeights + "/classifier_weights.h5")
model.compile(
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
#%% test time augmentation method
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
       # width_shift_range=0.2,
        #height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
        fill_mode='nearest')
         
predictions = []
for i in range(0,5):
    test_ds=train_datagen.flow_from_directory(
    dpTest,
    shuffle=True,    
)
    acc=model.evaluate_generator(test_ds, verbose=1)
    predictions.append(acc)
pred = np.mean(predictions, axis=0)
print("Test time augmentation method prediction accuracy:" +"%.1f%%" % (100 * pred[1]))
#%%
test_datagen = ImageDataGenerator(
        rescale=1./255,)
         
test_ds=test_datagen.flow_from_directory(
    dpTest,    
)
acc=model.evaluate(test_ds, verbose=0)
print("Prediction accuracy:" +"%.1f%%" % (100 * acc[1]))