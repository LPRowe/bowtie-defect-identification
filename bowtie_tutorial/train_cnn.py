# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 00:57:29 2021

@author: rowe1
"""

import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def get_combined_data():
    """
    Each row consists of comma separated values:
        standard deviation (std) of pixel values in shear 0 image, 
        std if pixels in shear 45 image,
        64 pixels values representing a flattened (8 by 8) cropped shear 0 image,
        64 pixel values representing a flattened (8 by 8) cropped shear 45 image,
        label (1 for bowtie, 0 for nonbowtie)
    """
    data = np.load('./training_data/data.npy')
    images, labels = [], []
    for row in data:
        images.append(np.reshape(row[2:-1], (16, 8)))
        labels.append(row[-1])
    return np.array(images), np.array(labels)
    

def split_data(X_data, y_data, train=0.8, test=0.1):
    """
    Splits the data into 3 sets:
        training data: to train the model
        validation data: to prevent overfitting of the model (used for early stopping)
        testing data: to test the final model on unseen data
        
    Set the fraction of the data that you wish to be allocated to training and testing.
    The remaining fraction will be allocated to validation.
    train + test < 1.0
    
    The recommended split (85, 10, 5) is skewed towards training because there are only 1000 samples.
    With more data, a healthier split might be train 64%, valid 16%, test 20%.
    
    returns 3 sets of data (Taining, Validation, Testing)
    Each dataset consists of (images, labels)
    """
    X_data, X_test, y_data, y_test = train_test_split(X_data, y_data, test_size=0.05, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=0.11, random_state=42)
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
    
    
# =============================================================================
# LOAD LABELED DATA
# =============================================================================
X_data, y_data = get_combined_data()
SHAPE = X_data[0].shape
MAP = {1: "Bowtie", 0: "Nonbowtie"}

# =============================================================================
# SPLIT DATA: training (85%), validation (10%) and testing (5%)
# split is skewed towards training because we only have ~1000 samples
# =============================================================================
(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = split_data(X_data, y_data)


# =============================================================================
# Plot the data to make sure that there are an even number of bowties and 
# nonbowties in each dataset
# =============================================================================
show_data_dist = True
if show_data_dist:
    plt.close('all')
    plt.figure(dpi=150)
    bar_width = 0.3
    plt.hist(y_train-bar_width, label = 'train', rwidth=bar_width)
    plt.hist(y_valid+bar_width, label = 'validation', rwidth=bar_width)
    plt.hist(y_test, label = 'test', rwidth=bar_width)
    plt.xlabel('Left: Non-bowties ; Right: Bowties')
    plt.ylabel('Count')
    plt.legend()

# =============================================================================
# BUILD MODEL
# Changes:
#   1. input shape is [16, 8, 1] for the shape of the image
#   2. output layer now has only 2 neurons (with only 2 neurons we do not need to use
#      a softmax activation function.  However, softmax provides us with prediction confidence
#      which can be used to ensemble the classifier with a second classifier if so desired)
# =============================================================================
model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=100, kernel_size=7, strides=1, padding='same', 
                              activation='relu', input_shape=[*SHAPE, 1])) # update the shape to match 
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(filters=100, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(filters=200, kernel_size=3, strides=2, padding='same'))
model.add(keras.layers.Conv2D(filters=200, kernel_size=3, strides=1, padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(2, activation='softmax'))  # Change our output to have 2 neurons (bowtie or not bowtie)

# =============================================================================
# COMPILE MODEL
# =============================================================================
optimizer = keras.optimizers.Nadam(lr=0.01, decay=0.001)
loss = keras.losses.sparse_categorical_crossentropy
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# =============================================================================
# SET CALLBACKS
# Early Stopping: avoids overfitting by stopping training when the accuracy of
#                 predictions made on the validation data set cease to improve
# Checkpoint: The final version of the model will be slightly overfit, so 
#             checkpoint automatically saves the version of the model that 
#             performed the best on the validation data
# =============================================================================
patience = 10 #how many epochs to wait without improvement before stopping
cb_earlystop = keras.callbacks.EarlyStopping(patience=patience)
cb_checkpoint = keras.callbacks.ModelCheckpoint('./best_bowtie_model.h5', save_best_only=True)
callbacks=[cb_earlystop, cb_checkpoint]

# =============================================================================
# ADD ONE EXTRA DIMENSION TO THE DATASETS (N, 16, 8, 1)
# =============================================================================
X_train = np.reshape(X_train, (len(X_train), *SHAPE, 1))
X_valid = np.reshape(X_valid, (len(X_valid), *SHAPE, 1))
X_test = np.reshape(X_test, (len(X_test), *SHAPE, 1))

# =============================================================================
# FIT MODEL
# batch_size: friends don't let friends train with a batch size > 32
# epochs: some large number, actual number of epochs will be limited by early stopping checkpoint
# X_train, y_train: dataset used to train the CNN
# X_val, y_val: data set to make predictions on between epochs to check if we are overfitting
# =============================================================================
h = model.fit(X_train, y_train, batch_size=32, epochs=1000, 
              callbacks=callbacks, validation_data=(X_valid, y_valid))

# =============================================================================
# PLOT HISTORY
# Plots the accuracy of predictions made on the training set and the accuracy
# of predictions made on the validation set.
# =============================================================================
plt.close('all')
epochs = h.epoch
plt.figure(dpi=150)
for m in h.history:
    if 'accuracy' in m:
        plt.plot(epochs,h.history[m], label=m)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# =============================================================================
# LOAD THE BEST MODEL
# =============================================================================
#best_model = keras.models.load_model('best_bowtie_model.h5')
best_model = model

# =============================================================================
# ASSESS THE MODEL ACCURACY ON THE UNSEEN TEST DATASET
# =============================================================================
correct = 0
for index in range(len(y_test)):
    pred = np.argmax(best_model.predict(np.reshape(X_test[index], (1, *SHAPE, 1))))
    correct += int(pred == y_test[index])
print(f"Accuracy: {int(100*correct/len(y_test))}%")
    
# =============================================================================
# OBSERVE THE MODEL AT WORK
# =============================================================================
def vis(fig_num = 0):
    """
    Display a random image from the test set along with the images actual and predicted labels.
    """
    index=np.random.randint(len(y_test))
    arr = np.reshape(X_test[index], (1, *SHAPE, 1))
    
    label = y_test[index]
    prediction = best_model.predict(arr)
    pred = np.argmax(prediction)
    confidence = prediction[0][pred]
    
    img = np.reshape(X_test[index], SHAPE)
    plt.figure(fig_num)
    plt.gray()
    plt.figure(dpi=150)
    plt.imshow(img)
    plt.title(f'Prediction: {MAP[pred]}\nActual: {MAP[label]}\nConfidence: {round(100*confidence, 1)}%')

for i in range(3):
    vis(fig_num = i)