import keras
import tensorflow as tf
import os
from os.path import join
import json
import random
import itertools
import re
import datetime
from collections import Counter
#import cairocffi as cairo
#import editdistance

import keras.callbacks
from keras import backend as K
from keras.models import load_model

import numpy as np
from scipy import ndimage
import pylab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2

from utils import *

train_dir = os.path.join('data','anpr_ocr__train')
test_dir = os.path.join('data','anpr_ocr__test')

c_val = load_json(train_dir, 'val')
c_train = load_json(train_dir, 'train')

letters_train = set(c_train.keys())
letters_val = set(c_val.keys())

if letters_train == letters_val:
    print('Letters in train and val do match')
else:
    raise Exception()
# print(len(letters_train), len(letters_val), len(letters_val | letters_train))
letters = sorted(list(letters_train))
print('Letters:', ' '.join(letters))

sess = tf.Session()
K.set_session(sess)

plate_images = TextImageGenerator(train_dir, 'val', 128, 64, 8, 4, letters)
plate_images.build_data()

if os.path.exists('model_1.h5'):
    print('True')
    model = train(128, train_dir, letters, load=True)
else:
    model = train(128, train_dir, letters, load=False)
    model.save('model_1.h5')

net_inp = model.get_layer(name='the_input').input
net_out = model.get_layer(name='softmax').output

model1 = load_model('model_1.h5', compile = False)

#model1.predict(cv2.imread('data\anpr_ocr__test\img/A007HA50.png'))
#model1.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    

for inp_value, _ in plate_images.next_batch():
    bs = inp_value['the_input'].shape[0]
    X_data = inp_value['the_input']
    net_out_value = sess.run(net_out, feed_dict={net_inp:X_data})
    pred_texts = decode_batch(net_out_value, letters)
    labels = inp_value['the_labels']
    texts = []
    for label in labels:
        text = ''.join(list(map(lambda x: letters[int(x)], label)))
        texts.append(text)
    
    for i in range(bs):
        fig = plt.figure(figsize=(10, 10))
        outer = gridspec.GridSpec(2, 1, wspace=10, hspace=0.1)
        ax1 = plt.Subplot(fig, outer[0])
        fig.add_subplot(ax1)
        fig.add_subplot(ax2)
        print('Predicted: %s\nTrue: %s' % (pred_texts[i], texts[i]))
        img = X_data[i][:, :, 0].T
        ax1.set_title('Input img')
        ax1.imshow(img, cmap='gray')
        ax1.set_xticks([])
        ax1.set_yticks([])
        #ax.axvline(x, linestyle='--', color='k')
        plt.show()
    break
