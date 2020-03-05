# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 23:00:16 2020

@author: weicc
"""
'''
Hide warning 
'''
import warnings
warnings.filterwarnings('ignore')

'''
impor Lib
'''
import numpy as np
import keras 
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools






train_path = 'train'
valid_path = 'valid'
test_path = 'test'

train_batches = ImageDataGenerator().flow_from_directory(train_path,target_size=(224,224),classes=['Wei','11'],batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path,target_size=(224,224),classes=['Wei','11'],batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(test_path,target_size=(224,224),classes=['Wei','11'],batch_size=10)

def plots(ims, rows=1, interp=False, titles=None):
    ims = np.array(ims).astype(np.uint8)
    if(ims.shape[-1]!=3):
        ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=(12,6))
    cols = len(ims)//rows if len(ims)%2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows,cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i],interpolation=None if interp else 'none')

def plot_confusion_maxtrix(cm, classes, 
                           normalize=False,
                           title='Confusion matrix',
                           cmap=plt.cm.Blues):
    
    """
    this function print nad plots the confustion matrix 
    Normalization can be applied by setting 'normalize=true'
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap= cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes,)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
        print("normalized confusion matrix")
    else :
        print("Confusion matrix, without normalization")
        
    print(cm)
    
    thresh = cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j],
                 horizontalalignment="center",
                 color="white" if cm[i,j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label') 


imgs, labels = next(train_batches)
plots(imgs,titles=labels)

'''
Build and Tranin simple CNN model
'''
'''
model = Sequential([
    Conv2D(32,(3,3), activation='relu', input_shape=(224,224,3)),
    Flatten(),
    Dense(2, activation='softmax'),
    ])

model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics =['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=1,
                    validation_data=valid_batches, validation_steps=1, epochs=1, verbose=1)


test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles = test_labels)

test_labels = test_labels[:,0]
test_labels

predictions = model.predict_generator(test_batches, steps=1, verbose=1)
predictions

cm = confusion_matrix(test_labels, predictions[:,0])
'''



#cm_plot_labels = ['11','Wei']
#plot_confusion_maxtrix(cm, cm_plot_labels, title='Confusion Matrix')

'''
Load v
gg16 model 
and using vgg16 model predict pic
'''

vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()

type(vgg16_model)

model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)
    
model.summary()

model.layers.pop()
model.summary()

for layer in model.layers:
    layer.trainable= False
    
model.add(Dense(2,activation='softmax'))
model.summary()

model.compile(Adam(lr=.00005), loss='categorical_crossentropy', metrics =['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=5,
                    validation_data=valid_batches, validation_steps=4, epochs=250, verbose=1)


test_imgs, test_labels = next(test_batches)
plots(test_imgs, titles = test_labels)

test_labels = test_labels[:,0]
test_labels

predictions = model.predict_generator(test_batches, steps=1, verbose=1)
predictions

cm = confusion_matrix(test_labels, np.round(predictions[:,0]))
cm_plot_labels = ['11','Wei']
plot_confusion_maxtrix(cm, cm_plot_labels, title='Confusion Matrix')

model.save('my_model.h5') 

model = load_model('my_model.h5')








