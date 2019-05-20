from keras.utils import np_utils
import sys
import keras
from keras import Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten, BatchNormalization, Activation, MaxPooling1D
from keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping
from keras.optimizers import Adam,SGD
import numpy as np
import os
import shutil
import time
from utilities import get_data 

dataset_path = '3_emotion'

print('Dataset path:',dataset_path)
print('Emotion:',os.listdir(dataset_path))
print('Num emotion:',len(os.listdir(dataset_path)))

x_train, x_test, y_train, y_test = get_data(dataset_path=dataset_path, max_duration = 4.0)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print('x_train:',x_train.shape)
print('y_train:',y_train.shape)

#create model
model = Sequential()

Ckernel_size = 3
Cstrides = 1
Ppool_size = 2
Pstrides = 2
padding = 'SAME'
acti = 'relu'

#CNN+LSTM
model.add(Conv1D(filters = 64, kernel_size = Ckernel_size, strides=Cstrides, padding=padding,input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(BatchNormalization())
model.add(Activation(activation = acti))
model.add(MaxPooling1D(pool_size=Ppool_size, strides=Pstrides,padding=padding))


model.add(Conv1D(filters = 64, kernel_size = Ckernel_size, strides=Cstrides, padding=padding))
model.add(BatchNormalization())
model.add(Activation(activation = acti))
model.add(MaxPooling1D(pool_size=Ppool_size*2, strides=Pstrides*2, padding=padding))


model.add(Conv1D(filters = 128, kernel_size = Ckernel_size, strides=Cstrides, padding=padding))
model.add(BatchNormalization())
model.add(Activation(activation = acti))
model.add(MaxPooling1D(pool_size=Ppool_size*2, strides=Pstrides*2, padding=padding))

model.add(Flatten())
model.add(Dense(1024))
model.add(Dense(y_train.shape[1], activation='softmax'))

#compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#save model to json file
model_json = model.to_json()
with open("model/model_json.json", "w") as json_file:
    json_file.write(model_json)

#callback funtion
tensorboard = TensorBoard('tensorboard/log_model_{}'.format(int(time.time())), write_graph=True , write_images=True)
checkpointer = ModelCheckpoint("./model/weights_best.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

print('===================================================')
print('Start trainning CNN model')
print('Run "tensorboard --logdir tensorboard" to view logs')
print('===================================================')

#fit
model.fit(x_train, y_train, batch_size=32, epochs=50, verbose=1, validation_split = 0.2, callbacks= [tensorboard,checkpointer])

#test model
print('Testing model')
loss,acc = model.evaluate(x_test, y_test)
print('===================================================')
print('Test Loss = {:.4f}'.format(loss))
print('Test Accu = {:.4f} %'.format(acc*100))
print('===================================================')
print("Train & Test done! Saved all to ./model for predict! ")
