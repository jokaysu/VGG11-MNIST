'''
$ python --version
Python 2.7.15

$ pip list
Package                       Version   
----------------------------- ----------
absl-py                       0.6.1     
astor                         0.7.1     
backports.functools-lru-cache 1.5       
backports.weakref             1.0.post1 
certifi                       2018.11.29
cycler                        0.10.0    
enum34                        1.1.6     
funcsigs                      1.0.2     
futures                       3.2.0     
gast                          0.2.0     
grpcio                        1.16.1    
h5py                          2.8.0     
Keras                         2.2.4     
Keras-Applications            1.0.6     
Keras-Preprocessing           1.0.5     
kiwisolver                    1.0.1     
Markdown                      3.0.1     
matplotlib                    2.2.3     
mock                          2.0.0     
numpy                         1.15.4    
pbr                           5.1.1     
Pillow                        5.3.0     
pip                           18.1      
protobuf                      3.6.1     
pyparsing                     2.3.0     
python-dateutil               2.7.5     
pytz                          2018.7    
PyYAML                        3.13      
scipy                         1.1.0     
setuptools                    39.0.1    
six                           1.11.0    
subprocess32                  3.5.3     
tensorboard                   1.12.0    
tensorflow                    1.12.0    
termcolor                     1.1.0     
Werkzeug                      0.14.1    
wheel                         0.32.3
'''

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras import optimizers, utils
from PIL import Image
import numpy as np
import tensorflow as tf
from PIL import ImageFilter
from keras import regularizers

#prep

x_size=6000 #switch comment to run 60000
#x_size=60000 #switch comment to run 60000

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train2 = np.ndarray((x_size, 32, 32))
x_test2 = np.ndarray((10000, 32, 32))

for i in [0, x_size-1]:
	im = Image.fromarray(x_train[i], mode='L')
	im = im.resize((32, 32))
	x_train2[i] = np.array(im)
for i in [0,x_test.shape[0]-1]:
	im = Image.fromarray(x_test[i], mode='L')
	im = im.resize((32, 32))
	x_test2[i] = np.array(im)

x_train2 = x_train2.reshape(x_train2.shape[0], 32, 32, 1)
x_test2 = x_test2.reshape(x_test2.shape[0], 32, 32, 1)

y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)
y_train2 = y_train[:6000]

#model

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
				 activation='relu', 
				 padding='same',
				 kernel_regularizer=regularizers.l2(0.01),
				 input_shape=(32, 32, 1),
				 data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
				 activation='relu', 
				 padding='same',
				 kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1),
				 activation='relu', 
				 padding='same',
				 kernel_regularizer=regularizers.l2(0.01)))
model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1),
				 activation='relu', 
				 padding='same',
				 kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1),
				 activation='relu', 
				 padding='same',
				 kernel_regularizer=regularizers.l2(0.01)))
model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1),
				 activation='relu', 
				 padding='same',
				 kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1),
				 activation='relu', 
				 padding='same',
				 kernel_regularizer=regularizers.l2(0.01)))
model.add(Conv2D(512, kernel_size=(3, 3), strides=(1, 1),
				 activation='relu', 
				 padding='same',
				 kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(4096, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))

#compile

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])

#fit

model.fit(x_train2, y_train2, epochs=1, batch_size=32) #switch comment to run 60000
#model.fit(x_train2, y_train, epochs=1, batch_size=32) #switch comment to run 60000

print('****************EVALUATE****************')

score = model.evaluate(x_train2, y_train2)
print('Train loss after 1 epoch:', score[0])
print('Train accuracy after 1 epoch:', score[1])

score = model.evaluate(x_test2, y_test)
print('Test loss after 1 epoch:', score[0])
print('Test accuracy after 1 epoch:', score[1])

model.fit(x_train2, y_train2, epochs=1, batch_size=32) #switch comment to run 60000
#model.fit(x_train2, y_train, epochs=1, batch_size=32) #switch comment to run 60000

score = model.evaluate(x_train2, y_train2)
print('Train loss after 2 epoch:', score[0])
print('Train accuracy after 2 epoch:', score[1])

score = model.evaluate(x_test2, y_test)
print('Test loss after 2 epoch:', score[0])
print('Test accuracy after 2 epoch:', score[1])

model.fit(x_train2, y_train2, epochs=1, batch_size=32) #switch comment to run 60000
#model.fit(x_train2, y_train, epochs=5, batch_size=32) #switch comment to run 60000

score = model.evaluate(x_train2, y_train2)
print('Train loss after 3 epoch:', score[0])
print('Train accuracy after 3 epoch:', score[1])

score = model.evaluate(x_test2, y_test)
print('Test loss after 3 epoch:', score[0])
print('Test accuracy after 3 epoch:', score[1])


print('****************ROTATION****************')

for i in range(-45, 50, 5):
	x_test3 = np.ndarray((10000, 32, 32))
	for j in [0,9999]:
		im = Image.fromarray(x_test[j], mode='L')
		im = im.resize((32, 32))
		im = im.rotate(i)
		x_test3[j] = np.array(im)
	x_test3 = x_test3.reshape(x_test3.shape[0], 32, 32, 1)
	score = model.evaluate(x_test3, y_test)
	print("Rotate ", i, " degrees, Accuracy: ", score[1])


print('****************BLURRING****************')

for i in range(0, 7):
	x_test4 = np.ndarray((10000, 32, 32))
	for j in [0,9999]:
		im = Image.fromarray(x_test[j], mode='L')
		im = im.resize((32, 32))
		im = im.filter(ImageFilter.GaussianBlur(i))
		x_test4[j] = np.array(im)
	x_test4 = x_test4.reshape(x_test4.shape[0], 32, 32, 1)
	score = model.evaluate(x_test4, y_test)
	print("Radius ", i, " degrees, Accuracy: ", score[1])
