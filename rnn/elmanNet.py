import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict


class model(object):

    def __init__(self):
        self.emb=[]
        self.Wx=[]
        self.Wh=[]
        self.W=[]
        self.bh=[]
        self.b=[]
        self.h0=[]


    def initRNN(self):

        # self.params = [self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ]
        # self.names  = ['Wx', 'Wh', 'W', 'bh', 'b', 'h0']

        # as many columns as context window size/lines as words in the sentence
        idxs = T.ivector()
        x = self.emb[idxs] #.reshape((idxs.shape[0], de * cs))
        y = T.iscalar('y')  # label

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
            # s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            return h_t

        h, _ = theano.scan(fn=recurrence, \
            sequences=x, outputs_info=[self.h0], \
            n_steps=x.shape[0])

        s = T.nnet.softmax(T.dot(h[-1,:], self.W) + self.b)

        p_y_given_x_lastword = s[-1,:]

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -T.mean(T.log(p_y_given_x_lastword)[y])

        gradients = T.grad(nll, self.params)
        
        updates = OrderedDict((p, p - lr * g)
                              for p, g in zip(self.params, gradients))

        # theano functions
        y_pred = T.argmax(p_y_given_x_lastword)

        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        self.p_cm = theano.function(
            inputs=[idxs], outputs=p_y_given_x_lastword)


        self.train = theano.function(inputs=[idxs, y, lr],
                                      outputs=nll,
                                      updates=updates)

        self.normalize = theano.function(inputs=[],
                         updates={self.emb:
                         self.emb / T.sqrt((self.emb ** 2).sum(axis=1)).dimshuffle(0, 'x')})

    def loadRNNModelFromFile(self,folder):
        # parameters of the model
        
        self.emb=theano.shared(numpy.load(os.path.join(folder, 'embeddings.npy'))
            .astype(theano.config.floatX))
        self.Wx=theano.shared(numpy.load(os.path.join(folder, 'Wx.npy'))
            .astype(theano.config.floatX))
        self.Wh=theano.shared(numpy.load(os.path.join(folder, 'Wh.npy'))
            .astype(theano.config.floatX))
        self.W=theano.shared(numpy.load(os.path.join(folder, 'W.npy'))
            .astype(theano.config.floatX))
        self.bh=theano.shared(numpy.load(os.path.join(folder, 'bh.npy'))
            .astype(theano.config.floatX))
        self.b=theano.shared(numpy.load(os.path.join(folder, 'b.npy'))
            .astype(theano.config.floatX))
        self.h0=theano.shared(numpy.load(os.path.join(folder, 'h0.npy'))
            .astype(theano.config.floatX))

        self.names = ['embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0']
        self.params = [self.emb, self.Wx, self.Wh, self.W, self.bh, self.b, self.h0]
        
        self.initRNN()

    def initRNNModelRand(self, nh, nc, ne, de):
        # parameters of the model
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size #delete
        '''
        # parameters of the model
        self.emb = theano.shared(0.2 *  numpy.random.uniform(-1.0, 1.0,\
                   (ne+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
        self.Wx  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (de, nh)).astype(theano.config.floatX))
        self.Wh  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nh)).astype(theano.config.floatX))
        self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (nh, nc)).astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
        self.h0  = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))

        # bundle
        self.params = [ self.emb, self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ]
        self.names  = ['embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0']
        
        self.initRNN()


    def saveModel(self, folder):
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())