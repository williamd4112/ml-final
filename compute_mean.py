import numpy as np
import sys

x = np.load(sys.argv[1])
x = np.mean(x, axis=0)
np.save(sys.argv[2], x)
print('Saved as %s' % sys.argv[2])

