import numpy
import time
import sys
import subprocess
import os
import random
import gzip
import cPickle

sys.path.append('../data')
sys.path.append('../rnn')

import elmanNet as Net

def trainCM(corpus,s):

    folder = s['folder']
    if not os.path.exists(folder):
        os.mkdir(folder)

    train_set=corpus[0]
    valid_set=corpus[1]
    test_set =corpus[2]
    dic = corpus[3]

    idx2label = dict((k, v) for v, k in dic['labels2idx'].iteritems())
    idx2word = dict((k, v) for v, k in dic['words2idx'].iteritems())

    train_lex, train_y = train_set
    valid_lex, valid_y = valid_set
    test_lex,  test_y = test_set

    vocsize=len(idx2word)
    nclasses=len(idx2label)

    nsentences = len(train_lex)

    # instanciate the model
    numpy.random.seed(s['seed'])
    random.seed(s['seed'])

    rnn=Net.model()
    rnn.initRNNModelRand(nh=s['nhidden'],
                nc=nclasses,
                ne=vocsize,
                de=s['emb_dimension'])

    # train with early stopping on validation set
    best_f1 = -numpy.inf
    s['clr'] = s['lr']

    for e in xrange(s['nepochs']):
        # shuffle
        shuffle([train_lex, train_y], s['seed'])
        s['ce'] = e

        s['clr'] = s['lr']
        tic = time.time()
        for i in xrange(nsentences):
            
            if s['decay']:
                s['clr']=s['lr']*float(nsentences-i)/float(nsentences)
            rnn.train(train_lex[i], train_y[i], s['clr'])

            rnn.normalize()

            print '[learning rate %f ]' % s['clr'],' epoch %i >> %2.2f%%' % (e, (i + 1) * 100. / nsentences), 'completed in %.2f (sec) <<\r' % (time.time() - tic),
            sys.stdout.flush()

        # evaluation 
        predictions_test = [rnn.classify(x) for x in test_lex]
        predictions_valid = [rnn.classify(x) for x in valid_lex]

        res_test = prf(predictions_test,test_y)
        res_valid = prf(predictions_valid,valid_y)

        if res_valid['f1'] > best_f1:

            best_f1 = res_valid['f1']

            print 'NEW BEST: epoch', e,'valid P:',res_valid['p'],'valid R:',res_valid['r'], 'valid F1', res_valid['f1'], 'best test F1', res_test['f1']
            s['vf1'], s['vp'], s['vr'] = res_valid[
                'f1'], res_valid['p'], res_valid['r']
            s['tf1'], s['tp'], s['tr'] = res_test[
                'f1'],  res_test['p'],  res_test['r']
            s['be'] = e

            rnn.saveModel(folder) # save mdoel

    print 'BEST RESULT: epoch', e, 'best valid F1', s['vf1'], 'best test F1', s['tf1'], 'with the model', folder

def prf(pre,truth):
    N=0.00001
    C=0.00001
    E=0.00001

    for p,t  in zip(pre,truth):
        
        if t==1:
            N+=1
        if p==1 and t==1:
            C+=1
        if p==1 and t==0:
            E+=1
    
    precision=C/(C+E)
    recall=C/N
    f1score=2*precision*recall/(precision+recall)

    return {'p':precision, 'r':recall, 'f1':f1score}

def shuffle(lol, seed):

    for l in lol:
        random.seed(seed)
        random.shuffle(l)

if __name__ == '__main__':
    f=gzip.open('/home/gy/theano/rnn/is13/word.corpus/corpus2.mini.pkl.gz','rb')
    corpus=cPickle.load(f)
    f.close()
    s = {'lr': 0.05,
         'decay': True,  # decay on the learning rate 
         'nhidden': 100,  # number of hidden units
         'seed': 345,
         'emb_dimension': 100,  # dimension of word embedding
         'nepochs': 50,
         'folder':'model'} # model saved folder
    trainCM(corpus,s)
    