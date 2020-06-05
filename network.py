import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import random
import pickle
import time

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)

BS = 4
EPOCHS = 40

IMG_HEIGHT = 512
IMG_WIDTH = 512
CATEGORIES = ["flip", "nothing"]
# CATEGORIES = ["up", "down", "left", "right", "flip"]
NAME = "telloNetCNNNewDataset416Gray_1class-{}".format(int(time.time()))
#
# ############ CREATE DATASET ######################
# DATADIR = "C:\\Users\\andre\\Documents\\TELLO\\telloDetect5train"
# print(DATADIR)
# training_data = []
#
# def create_training_data():
#     for category in CATEGORIES:
#         path = os.path.join(DATADIR, category)
#         class_num = CATEGORIES.index(category)
#         for img in os.listdir(path):
#             try:
#                 img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#                 img_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
#                 # cv2.imshow(str(class_num), img_array)
#                 # cv2.waitKey()
#                 training_data.append([img_array, class_num])
#             except Exception as e:
#                 pass
#
# create_training_data()
#
# print("training data length: ", len(training_data))
#
# random.shuffle(training_data)
#
# X = []
# y = []
#
# for features, label in training_data:
#     X.append(features)
#     y.append(label)
#
# X = np.array(X).reshape(-1, IMG_WIDTH, IMG_HEIGHT, 1)
# ############ SAVE DATASET ######################
# pickle_out = open("x_train1Gray.pickle", "wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()
#
# pickle_out = open("y_train1Gray.pickle", "wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()

# ############ LOAD DATASET ######################
# pickle_in = open("x_train1Gray.pickle", "rb")
# X = pickle.load(pickle_in)
#
# pickle_in = open("y_train1Gray.pickle", "rb")
# y = pickle.load(pickle_in)
# #
pickle_in = open("x_testGray.pickle", "rb")
X_test = pickle.load(pickle_in)

pickle_in = open("y_testGray.pickle", "rb")
y_test = pickle.load(pickle_in)
# #
# # # ############ NORMILIZE TRAINING DATA ######################
# X = X.astype('float32') / 255
# y = keras.utils.to_categorical(y)
# #
X_test = X_test.astype('float32') / 255
y_test = keras.utils.to_categorical(y_test)
#
# print(X.shape[1:])
#
# # https://keras.io/preprocessing/image/
# # Add augmentations if needed
# # An image generator can perform data augmentation
# # Currently the image generator does not make any augmentations
#
# train_datagen = ImageDataGenerator(featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     vertical_flip=True)
#
# # Define the iterator to provide us with data and labels from the image
# # genereator during training
# it_train = train_datagen.flow(X, y, batch_size=BS)
#
# # ############ BUILD MODEL ######################
# # https://www.tensorflow.org/guide/keras/train_and_evaluate
# tf.keras.backend.clear_session()
#
#
# # ***
# # Start-Edit (Mandatory)
# # TODO: Create an improved deep net by making sensible architecture and implementation choices
#
# def conv_block2(input_data, filters, conv_size):
#     # Define a block that can be reused:
#     # input_data: The input tensor i.e., the previous layer
#     # filters: number of filters in the convolutional layer
#     # conv_size: The size of the kernel
#
#     # 2D convolutional layer with stride = 2
#     # also adds padding so the input and output size do not change with respect
#     # when filter is applied outside of the image
#     x = layers.Conv2D(filters, conv_size, activation='linear', padding='same', strides=2)(input_data)
#     # x = layers.Conv2D(filters, conv_size, activation='linear', padding='same', strides=1)(x)
#     # Applying element wise relu activation
#     x = layers.Activation('relu')(x)
#     # normilizing layers after the dual conv layers and the activation layer
#     # x = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(x)
#     # Add Dropout reqularization layer with rate 0.2
#     # x = layers.Dropout(rate=0.05, noise_shape=None, seed=None)(x)
#     # return the last layer of the block
#     return x
#
#
# # Constructs the computational graph corresponding to the neural network
# # Uses the Keras Functional API https://keras.io/getting-started/functional-api-guide/
# # Determine the input to the model
# inputs = keras.Input(shape=X.shape[1:])
# # Define a block of two convolution+activation with 128 filters and kernel size 3x3
# x = conv_block2(inputs, filters=128, conv_size=3)
# # Define a block of two convolution+activation with 128 filters and kernel size 3x3
# # x = conv_block2(x, filters=256, conv_size=3)
# # Define a block of two convolution+activation with 128 filters and kernel size 3x3
# # x = conv_block2(x, filters=256, conv_size=3)
# # Add MaxPooling layer
# x = layers.MaxPooling2D((2,2))(x)
# # Define a block of two convolution+activation with 64 filters and kernel size 3x3
# x = conv_block2(x, filters=128, conv_size=3)
# # Define a block of two convolution+activation with 64 filters and kernel size 3x3
# x = conv_block2(x, filters=64, conv_size=3)
# # Add MaxPooling layer
# x = layers.MaxPooling2D((2,2))(x)
# # Vectorize the input to 1D
# x = layers.Flatten()(x)
# # # Add a dense layer with 400 neurons and relu activation
# # x = layers.Dense(400, activation='relu')(x)
# # Add a dense layer with 200 neurons and relu activation
# # x = layers.Dense(200, activation='relu')(x)
# # # Add a dense layer with 100 neurons and relu activation
# # x = layers.Dense(100, activation='relu')(x)
#
# # Stop-Edit
# # ***
#
# # Add a dense layer with 5 neurons, one for each class
# x = layers.Dense(2, activation='linear')(x)
# # Pass through an activation function to normalize probabilities
# outputs = layers.Activation('sigmoid')(x)
#
# # Construct the model
# base_net = keras.Model(inputs, outputs)
#
# # Set the optimizer Adam with learning rate 0.001, and the loss function the
# # categorical cross-entropy, ALso monitor the accuracy
# base_net.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss=losses.binary_crossentropy, metrics=['acc'])
#
# # Print the model
# base_net.summary()
#
# ############ ADD TRAINING CALLBACKS ######################
# mchkp = keras.callbacks.ModelCheckpoint('./{}.h5'.format(NAME), monitor='val_acc', save_best_only=True, save_weights_only=False)
# # tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/{}'.format(NAME))
# callbacks = [mchkp]#, tensorboard]
#
# ############ TRAIN AND SAVE MODEL ######################
# base_net.fit_generator(generator = it_train,epochs=EPOCHS,validation_data=(X_test,y_test),shuffle=True,steps_per_epoch=len(X) // BS,callbacks=callbacks)
#
# base_net.save("./{}.model".format(NAME))
#
# # ############ LAOD MODEL ######################
# model = keras.models.load_model('./{}.h5'.format(NAME))
#
model = keras.models.load_model("./telloNetCNNNewDataset512Gray_2class-1590751265 (1).h5")
model.summary()

print('\nEvaluating on test data')
results = model.evaluate(X_test, y_test, batch_size=BS)
print('test loss, test acc:', results)
#
# model = keras.models.load_model('./{}.model'.format(NAME))
# print('\nEvaluating Model on test data')
# results = model.evaluate(X_test, y_test, batch_size=BS)
# print('test loss, test acc:', results)

# # ############ LOAD AND PRE-PROCESS TEST IMAGE ######################
# test_img= cv2.imread("C:\\Users\\andre\\Documents\\TELLO\\telloDetect5train\\down\\229.jpeg", cv2.IMREAD_COLOR)
# test_img = cv2.resize(test_img, (IMG_WIDTH, IMG_HEIGHT))
# cv2.imshow("test", test_img)
# cv2.waitKey(2)
#
# test_img = test_img.astype('float32') / 255
# test_img = test_img.reshape(-1, IMG_WIDTH, IMG_HEIGHT, 3)
# #
# # ############ GET AND PRINT PREDICTION ######################
# prediction = model.predict([test_img])
#
# pred_name = CATEGORIES[np.argmax(prediction)]
# print(pred_name)
#
# print(prediction)
# print(np.argmax(prediction))