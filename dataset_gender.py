import cv2
import numpy as np
import csv

import sys, os
from tqdm import *

from os import listdir
from os.path import isfile, join

import argparse
import logging

import cPickle as pickle

AGES = ['child', 'young', 'adult', 'elder']
GENDERS = ['male', 'female']

IMG_DIM = (224, 224)

class Dataset():
    def load_labeled_data(self, raw_data_path):
        X = []
        T = []
        for g_i in range(len(GENDERS)):
            for a_i in range(len(AGES)):
                age = AGES[a_i]
                gender = GENDERS[g_i]
                label = g_i
                label = np.array([label], dtype=np.int32)[np.newaxis, :]
                
                path =  join(raw_data_path, age, gender)
                onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
                onlyfiles.sort(key=lambda f: int(filter(str.isdigit, f)))
 
                for i in tqdm(range(len(onlyfiles))):
                    img = (cv2.imread(path + '/' + onlyfiles[i]))
                    img = cv2.resize(img, IMG_DIM)
                    X.append(img[np.newaxis, :])
                    T.append(label)
        X = np.asarray(np.concatenate(X))
        T = np.asarray(np.concatenate(T))
        return X, T

    def load_unlabeled_data(self, raw_data_path):
        X = []
        F = []
        for g_i in range(len(GENDERS)):
            for a_i in range(len(AGES)):
                age = AGES[a_i]
                gender = GENDERS[g_i]
                label = g_i
                label = np.array([label], dtype=np.int32)[np.newaxis, :]
                
                path =  join(raw_data_path, age, gender)
                onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
                onlyfiles.sort(key=lambda f: int(filter(str.isdigit, f)))
 
                for i in tqdm(range(len(onlyfiles))):
                    img = cv2.cvtColor(cv2.imread(join(path, onlyfiles[i]), cv2.COLOR_RGB2GRAY))
                    img = cv2.resize(img, IMG_DIM)
                    X.append(img[np.newaxis, :])
                    F.append(onlyfiles[i])
        X = np.asarray(np.concatenate(X))
        return X, F

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='load labeled/unlabeled data', 
            choices=['label', 'unlabel'], type=str, default='label') 
    parser.add_argument('--input', help='input', type=str, default='data')
    parser.add_argument('--output', help='output', type=str, default='data')
    
    args = parser.parse_args()
    dataset = Dataset()

    if args.type == 'label':
        X, T = dataset.load_labeled_data(args.input)
        np.save(join(args.output, 'X'), X)
        np.save(join(args.output, 'T'), T)
    elif args.type == 'unlabel':
        X, F = dataset.load_unlabeled_data(args.input)
        np.save(join(args.output, 'X'), X)
        pickle.dump(F, open(join(args.input, 'X.index'), "wb"), True) 
        
                              
