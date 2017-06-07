import cv2
import numpy as np
import keras
import csv

import sys, os
from tqdm import *

import argparse
import logging
import tensorflow as tf
 
from model import *
from preprocess import *

from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from keras import backend as K

def load_train_dataset(x_train_path, t_train_path, x_test_path, t_test_path):
    def _load(x_path, t_path):
        X = np.load(x_path)
        T = np.load(t_path)
        return X, keras.utils.to_categorical(T, T.max() + 1)
    x_train, t_train = _load(x_train_path, t_train_path)
    x_test, t_test = _load(x_test_path, t_test_path)
    return x_train.astype(np.float32), t_train, x_test.astype(np.float32), t_test

def get_model(archi, X_shape, T_shape):
    if archi == 'vgg16':
        return VGG16Model(num_classes=T_shape[1], shape=X_shape[1:])
    elif archi == 'vgg19':
        return VGG19Model(num_classes=T_shape[1], shape=X_shape[1:])
    else:
        raise NotImplemented()

def main(args):
    # Config session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # Load dataset
    logging.info('Load dataset')
    X_train, T_train, X_test, T_test = load_train_dataset(args.X_train, args.T_train, args.X_test, args.T_test)
    
    # Preprocess dataset
    logging.info('Preprocess data')
    logging.info('X_train = %s, T_train = %s' % (X_train.shape, T_train.shape))
    logging.info('X_test = %s, T_test = %s' % (X_test.shape, T_test.shape))
    scale = 1.0 / 255.0
    mean = np.load(args.mean).astype(np.float32)
    X_train = pre_process(X_train, mean, scale)
    X_test = pre_process(X_test, mean, scale)

    # Setup model
    logging.info('Setup model')
    model = get_model(args.archi, X_train.shape, T_train.shape)
    
    # Training
    if args.ckpt != None:
        logging.info('Reload from %s' % args.ckpt)
        model.model = load_model(args.ckpt)
    model.train(X_train, T_train, X_test, T_test, logdir=args.logdir, batch_size=args.batch_size, epochs=args.epoch, lr=args.lr, augment=bool(args.augment))

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--X_train', help='X_train', type=str, required=True)
    parser.add_argument('--T_train', help='T_train', type=str, required=True)
    parser.add_argument('--X_test', help='X_test', type=str, required=True)
    parser.add_argument('--T_test', help='T_test', type=str, required=True)
    parser.add_argument('--mean', help='mean', type=str, required=True)
    parser.add_argument('--logdir', help='model log dir', type=str, required=True)
    parser.add_argument('--lr', help='lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('--epoch', help='epoch', type=int, default=10)
    parser.add_argument('--ckpt', help='ckpt', type=str)
    parser.add_argument('--archi', help='model name', type=str, default='vgg16')
    parser.add_argument('--augment', help='augment', type=int, default=1)
    parser.add_argument('--pre', help='pretrain model', type=str, default=None)
    args = parser.parse_args()
    main(args)
 
