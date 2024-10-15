import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from keras import backend as K
from sklearn.metrics import confusion_matrix,classification_report
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train[:5000,:,:]
x_test=x_test[:1000,:,:]
y_train=y_train[:5000]
y_test=y_test[:1000]
x_train=x_train/255
x_test=x_test/255
x_train=x_train.reshape(5000,28,28,1)
x_test=x_test.reshape(1000,28,28,1)
#CNN Architecture
m=keras.Sequential()
m.add(Conv2D(128,(3,3),activation='relu',input_shape=(28,28,1)))
m.add(MaxPooling2D(pool_size=(2,2)))
m.add(Conv2D(64,(3,3),activation='relu'))
m.add(MaxPooling2D(pool_size=(2,2)))
m.add(Conv2D(32,(3,3),activation='relu'))
m.add(MaxPooling2D(pool_size=(2,2)))
m.add(Flatten())
m.add(Dense(128,activation='relu'))
m.add(Dense(10,activation='softmax'))
m.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
h=m.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=20)
r=pd.DataFrame(h.history)
r['Epochs']=h.epoch
print(r.tail())
#Predictions on test_data
ypred=m.predict(x_test)
print(ypred)
ypred_m=[np.argmax(i) for i in ypred]
print(ypred_m[:10])
print(list(y_test[:10]))
#Confusion matrix
cm=confusion_matrix(y_test,ypred_m)
print("Confusion matrix")
print(cm)
#Classification Report
print("Classification Report")
print(classification_report(y_test,ypred_m))
#loss vs val_loss depiction
plt.subplot(1,2,1)
plt.plot(r['Epochs'],r['loss'],label='Training Loss')
plt.plot(r['Epochs'],r['val_loss'],label='Testing Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
#accuracy vs val_accuracy depiction
plt.subplot(1,2,2)
plt.plot(r['Epochs'],r['accuracy'],label='Training Accuracy')
plt.plot(r['Epochs'],r['val_accuracy'],label='Testing Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.show()



m.save('model.h5')


