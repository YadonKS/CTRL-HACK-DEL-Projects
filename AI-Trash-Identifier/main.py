
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import models

base_dir = './data2/DATASET'
train_dir = os.path.join(base_dir, 'TRAIN')
validation_dir = os.path.join(base_dir, 'TEST')

# Directory with training organic pictures
train_organic_dir = os.path.join(train_dir, 'O')

# Directory with training recyclable pictures
train_recycle_dir = os.path.join(train_dir, 'R')

# Directory with validation organic pictures
validation_organic_dir = os.path.join(validation_dir, 'O')

# Directory with validation recyclable pictures
validation_recycle_dir = os.path.join(validation_dir, 'R')

print('total training organic images:', len(os.listdir(train_organic_dir)))
print('total training recyclable images:', len(os.listdir(train_recycle_dir)))
print('total validation organic images:', len(os.listdir(validation_organic_dir)))
print('total validation recyclable images:', len(os.listdir(validation_recycle_dir)))

train_organic_fnames = os.listdir(train_organic_dir)
train_recycle_fnames = os.listdir(train_recycle_dir)
train_recycle_fnames.sort()

# %matplotlib inline

# Parameters for our graph; we'll output images in a 2x4 configuration
nrows = 2
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 2x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 4
next_organic_pix = [os.path.join(train_organic_dir, fname)
                    for fname in train_organic_fnames[pic_index - 4:pic_index]]
next_recycle_pix = [os.path.join(train_recycle_dir, fname)
                    for fname in train_recycle_fnames[pic_index - 4:pic_index]]

for i, img_path in enumerate(next_organic_pix + next_recycle_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')  # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)
    plt.imshow(img)
    print(img.shape)

plt.show()

(188, 269, 3)
(164, 308, 3)
(197, 256, 3)
(225, 225, 3)
(160, 314, 3)
(150, 300, 3)
(187, 269, 3)
(168, 300, 3)

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,               # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        shuffle=True,
        class_mode='binary')

# Flow validation images in batches of 20 using val_datagen generator
validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        shuffle=False,
        class_mode='binary')

# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
# the three color channels: R, G, and B
img_input = layers.Input(shape=(150, 150, 3))

# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Third convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Flatten feature map to a 1-dim tensor so we can add fully connected layers
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(1, activation='sigmoid')(x)

# # Create model:
# # input = input feature map
# # output = input feature map + stacked convolution/maxpooling layers + fully
# # connected layer + sigmoid output layer
# model = Model(img_input, output)
#
# model.compile(loss='binary_crossentropy',
#               optimizer=RMSprop(lr=0.001),
#               metrics=['acc'])
#
# history = model.fit(
#       train_generator,
#       epochs=8,
#       validation_data=validation_generator,
#       verbose=2,
#       shuffle=True)


model = models.load_model('model.h5')

# #@title Graphing accuracy and loss
# # Retrieve a list of accuracy results on training and validation data
# # sets for each training epoch
# acc = history.history['acc']
# val_acc = history.history['val_acc']
#
# # Retrieve a list of list results on training and validation data
# # sets for each training epoch
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# # Get number of epochs
# epochs = range(len(acc))
#
# # Plot training and validation accuracy per epoch
# plt.plot(epochs, acc, label='training accuracy')
# plt.plot(epochs, val_acc, label='validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend()
#
# # Plot training and validation loss per epoch
# plt.figure()
# plt.plot(epochs, loss, label='training loss')
# plt.plot(epochs, val_loss, label='validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# # loading in the Inception v3 model
# !wget --no-check-certificate \
#     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
#     -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
#
# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.optimizers import SGD
#
# local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
# pre_trained_model = InceptionV3(
#     input_shape=(150, 150, 3), include_top=False, weights=None)
# pre_trained_model.load_weights(local_weights_file)
#
#
# for layer in pre_trained_model.layers:
#   layer.trainable = False
#
#
# last_layer = pre_trained_model.get_layer('mixed7')
# print('last layer output shape:', last_layer.output_shape)
# last_output = last_layer.output

# # Flatten the output layer to 1 dimension
# x = layers.Flatten()(last_output)
# # Add a fully connected layer with 1,024 hidden units and ReLU activation
# x = layers.Dense(1024, activation='relu')(x)
# # Add a dropout rate of 0.5
# x = layers.Dropout(0.5)(x)
# # Add a final sigmoid layer for classification
# x = layers.Dense(1, activation='sigmoid')(x)
#
# unfreeze = False
#
# # Unfreeze all models after "mixed6"
# for layer in pre_trained_model.layers:
#   if unfreeze:
#     layer.trainable = True
#   if layer.name == 'mixed6':
#     unfreeze = True
#
# # Configure the model
# model = Model(pre_trained_model.input, x)
#
# # As an optimizer, here we will use SGD
# # with a very low learning rate (0.00001)
# model.compile(loss='binary_crossentropy',
#               optimizer=SGD(
#                   lr=0.00001,
#                   momentum=0.9),
#               metrics=['acc'])
#
# model.summary()

# Flow validation images using val_datagen generator
val_visual = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        class_mode='binary',
        shuffle=False
)

print(val_visual.class_indices)

val_pred_prob = model.predict(val_visual)

# must get index array before getting predictions!
val_dir_indices = val_visual.index_array
val_true_labels = [0 if n < 1112 else 1 for n in val_dir_indices] # directory is sorted alphanumerically; 1st 1112: 'O', 2nd 1112: 'R'

# getting predictions in the form of probablities
val_pred_prob = model.predict(val_visual)

# converting the probablities into binary values
val_pred_labels = [1 if n >= 0.5 else 0 for n in val_pred_prob]

print("Model predictions: "+str(val_pred_labels))
print("Actual labels:     "+str(val_true_labels))

# determining the filepaths of misclassified waste
num_misclasssified = 0
misclassified_filepaths = []
correctness = []
for pred_label, true_label, dir_index in zip(val_pred_labels, val_true_labels, val_visual.index_array):
  misclassified_filepaths.append(val_visual.filepaths[dir_index])
  if pred_label != true_label:
    correctness.append('incorrect')
    num_misclasssified += 1
  else:
    correctness.append('correct')

print("# of total images: "+str(len(correctness)))
print("# of misclassified images: "+str(num_misclasssified))

# model.save('model.h5')

