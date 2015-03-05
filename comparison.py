import numpy as np
import time

import python_recursive as r
import python_iterative as i

rng = np.random.RandomState(1234)
incomingProbabilities = rng.random_sample((100, 30))*10e-2        # random incoming probabilities for 4 time steps
sequence = [0]


recursiveCTC = r.RecursiveCTCLayer()
d1 = time.time()
recursive = recursiveCTC.ctc(incomingProbabilities, sequence)
d1 = time.time() - d1

iterativeCTC = i.IterativeCTCLayer()
d2 = time.time()
iterative = iterativeCTC.ctc(incomingProbabilities, sequence)
d2 = time.time() - d2


print 'Recursive:', recursive, '\tTime: %f' % (d1)
print 'Iterative:', iterative, '\tTime: %f' % (d2)