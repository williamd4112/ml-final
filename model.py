from vgg16 import VGG16
from keras import optimizers
from sklearn.model_selection import KFold

class VGG16Model(object):
    def __init__(self, num_classes=8, weight=None):
        self.num_classes = num_classes
        self.model = VGG16(classes=num_classes, pooling='max', weights=None) 
    
    def train(self, X, T, fold=3, batch_size=2, epochs=10):
        sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        kf = KFold(n_splits=fold)
        for train, test in kf.split(X):
            self.model.fit(X[train], T[train],
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X[test], T[test]),
                shuffle=True)

  
         
