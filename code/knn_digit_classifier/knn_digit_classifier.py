#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 18:59:19 2021

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

X = np.load('pixels.npy')
y = np.load('labels.npy')
newX = np.load('new_pixels.npy')

def knn_digit_classifier(k, digit):
    distances = np.sum((X-digit)**2,axis=1)
    
    knn = np.argsort(distances)[0:k]
    
    proba = np.bincount(y[knn],minlength=10)/k
    
    fig,ax = plt.subplots(1, 2, figsize = (15,5), gridspec_kw={'width_ratios':[1,2]})
    
    ax[0].imshow(digit.reshape(28,28),cmap="gray")
    ax[0].axis(False)
    
    ax[1].bar(x=np.arange(0,10), height=proba)
    ax[1].set_ylabel('probabilities', fontsize=15)
    ax[1].set_xticks(np.arange(0,10))
    
    plt.show()
    
knn_digit_classifier(100, newX[100])