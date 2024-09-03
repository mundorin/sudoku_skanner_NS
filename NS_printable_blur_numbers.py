import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import load_model
import cv2
import glob
test=700
train=1400
masx=[]
masy=[]
path_NS='LNS/inttranslated_printable_blur.h5'
for i in range(train):
    path = glob.glob('shot-' + str(i) + '-#?#.png', root_dir='C:\python 3.9 programs\sudoku_solver\learn_set_gaussen_noize')
    imgn = cv2.imread('learn_set_gaussen_noize/'+path[0],cv2.COLOR_BGR2GRAY)
    masx.append(imgn)
    masy.append(int(path[0][-6]))
masx=np.array(masx)
masy=np.array(masy)
(x_train, y_train) = masx,masy
masx=[]
masy=[]
for i in range(test):
    path = glob.glob('shot-' + str(i) + '-#?#.png', root_dir='C:\python 3.9 programs\sudoku_solver\learn_set_test')
    imgn = cv2.imread('learn_set_test/'+path[0],cv2.COLOR_BGR2GRAY)
    masx.append(imgn)
    masy.append(int(path[0][-6]))
masx=np.array(masx)
masy=np.array(masy)
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_test, y_test) = masx,masy

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# отображение первых 25 изображений из обучающей выборки
plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i+25], cmap=plt.cm.binary)

plt.show()

print('если хотите обучить новую нажмите 1 если использовать существующею 2')
x=input()
if x=='1':
    model = keras.Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
else:
    model=load_model(path_NS)
print(model.summary())      # вывод структуры НС в консоль

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


model.fit(x_train, y_train_cat, batch_size=32, epochs=20, validation_split=0.2)

model.evaluate(x_test, y_test_cat)

n = 1
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
print( res )
print( np.argmax(res),'hhgfh' )

plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

# Распознавание всей тестовой выборки
pred = model.predict(x_test)
pred = np.argmax(pred, axis=1)

print(pred.shape)

print(pred[:20])
print(y_test[:20])

# Выделение неверных вариантов
mask = pred == y_test
print(mask[:10])

x_false = x_test[~mask]
y_false = x_test[~mask]

print(x_false.shape)

# Вывод первых 25 неверных результатов
if len(x_false)>0:
    plt.figure(figsize=(10,5))
    for i in range(min(len(x_false),25)):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_false[i], cmap=plt.cm.binary)

plt.show()
model.save(path_NS)
