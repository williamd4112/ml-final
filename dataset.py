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
import glob

from sklearn.model_selection import KFold, train_test_split
from preprocess import FaceDetector

AGES = ['child', 'young', 'adult', 'elder']
GENDERS = ['male', 'female']

class Dataset():
    def __init__(self, label_type, size, crop, color):
        # label_type: gender, age, mix
        # size: 80x80
        # crop: is detect face (0, 1)
        # color: 'rgb', 'gray'
        # mean: image mean
        self.label_type = label_type
        self.dim = (size, size)
        self.crop = crop
        self.color = color
        self.face_detector = FaceDetector()

    def _get_label(self, g_i, a_i):
        if self.label_type == 'gender':
            return g_i
        elif self.label_type == 'age':
            return a_i
        elif self.label_type == 'mix':
            return g_i * len(AGES) + a_i
        else:
            assert False, self.label_type

    def _process_img(self, img):
        if self.crop:
            img = self.face_detector.detect(img)
        if self.color == 'gray':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _load_img(self, path):
        img = (cv2.imread(path))
        img = self._process_img(img)
        img = cv2.resize(img, self.dim)
        return img

    def load_labeled_data(self, raw_data_path, ext='*.jpg'):
        X = []
        T = []
        for g_i in range(len(GENDERS)):
            for a_i in range(len(AGES)):
                age = AGES[a_i]
                gender = GENDERS[g_i]
                label = self._get_label(g_i, a_i)
                label = np.array([label], dtype=np.int32)[np.newaxis, :]
                
                dir =  os.path.join(raw_data_path, age, gender)
                for filename in tqdm(glob.glob(os.path.join(dir, ext))):
                    img = self._load_img(filename)
                    X.append(img[np.newaxis, :])
                    T.append(label)
        X = np.asarray(np.concatenate(X))
        T = np.asarray(np.concatenate(T))
        return X, T

    def __call__(self, dir, ext='*.jpg'):
        for filename in sorted(glob.glob(os.path.join(dir, ext))):
            img = self._load_img(filename)
            yield filename, img

if __name__ == '__main__':
    '''
    Process training data from images in directory
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', help='label type', 
            choices=['gender', 'age', 'mix'], type=str, required=True) 
    parser.add_argument('--input', help='input', type=str, default='data')
    parser.add_argument('--output', help='output', type=str, default='dataset')
    parser.add_argument('--size', help='size', type=int, default=80)
    parser.add_argument('--crop', help='crop face', type=int, default=1)
    parser.add_argument('--frac', help='test data fraction', type=float, default=0.2)
    parser.add_argument('--color', help='color mode', type=str, default='rgb')    
    args = parser.parse_args()

    dataset = Dataset(label_type=args.label, size=args.size, crop=bool(args.crop), color=args.color)

    out_x_train = '_'.join(['X_train', 'crop', args.color, str(args.crop), args.label, str(args.size)])
    out_t_train = '_'.join(['T_train', 'crop', args.color, str(args.crop), args.label, str(args.size)])
    out_x_test = '_'.join(['X_test', 'crop', args.color, str(args.crop), args.label, str(args.size)])
    out_t_test = '_'.join(['T_test', 'crop', args.color, str(args.crop), args.label, str(args.size)])

    X, T = dataset.load_labeled_data(args.input)
    X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=args.frac)

    np.save(join(args.output, out_x_train), X_train)
    np.save(join(args.output, out_t_train), T_train)
    np.save(join(args.output, out_x_test), X_test)
    np.save(join(args.output, out_t_test), T_test) 
                              
