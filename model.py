from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import os 

from keras import optimizers
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import KFold, train_test_split

from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K

from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape


class PredictionModel(object):
    def train(self, X_train, T_train, X_test, T_test, logdir, fold=3, batch_size=64, epochs=100, lr=1e-5, augment=False):
        # Setup optmizer
        opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-9)
        self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        # Setup checkpoint
        filepath = os.path.join(logdir, "model-{epoch:02d}-{val_acc:.2f}.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')    
        tensorboard = TensorBoard(log_dir=logdir, histogram_freq=1)

        if augment:
            train_datagen = ImageDataGenerator(
                    rotation_range=90,
                    shear_range=0.2,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=True)
            train_datagen.fit(X_train)
            test_datagen = ImageDataGenerator()

            self.model.fit_generator(
                            train_datagen.flow(X_train, T_train, batch_size=batch_size, shuffle=True),
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            epochs=epochs,
                            verbose=True,
                            validation_data=test_datagen.flow(X_test, T_test, batch_size=batch_size),
                            validation_steps=X_test.shape[0] // batch_size,
                            callbacks=[checkpoint, tensorboard])
        else:
            self.model.fit(X_train, T_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(X_test, T_test),
                  shuffle=True,
                  callbacks=[checkpoint])

class VGG16Model(PredictionModel):
    def __init__(self, num_classes, shape, weight=None):
        self.num_classes = num_classes
        input_tensor = Input(shape=shape)
        base_model = VGG16(input_tensor=input_tensor, classes=num_classes, pooling='max', weights=weight, include_top=False)
        x = base_model.output
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(num_classes, activation='softmax', name='predictions')(x) 
        self.model = Model(inputs=base_model.input, outputs=x)  

class VGG19Model(PredictionModel):
    def __init__(self, num_classes, shape, weight=None):
        self.num_classes = num_classes
        input_tensor = Input(shape=shape)
        base_model = VGG19(input_tensor=input_tensor, classes=num_classes, pooling='max', weights=weight, include_top=False)
        x = base_model.output
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(num_classes, activation='softmax', name='predictions')(x) 
        self.model = Model(inputs=base_model.input, outputs=x)  
