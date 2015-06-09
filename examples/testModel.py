import sys
import cPickle

sys.path.append('../rnn')

import elmanNet as Net

model=Net.model()
model.loadRNNModelFromFile('model')

word=[1,2,3,4]
print model.p_cm(word)
print model.classify(word)
