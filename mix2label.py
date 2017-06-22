import numpy as np
import sys, os

if __name__ == '__main__':
    data_path = sys.argv[1]
    label = sys.argv[2]
    out_path = sys.argv[3]
    
    data = np.load(data_path)
    
    if label == 'gender':
        data = data % 2
    elif label == 'age':
        data = data / 4

    np.save(out_path, data)
    
