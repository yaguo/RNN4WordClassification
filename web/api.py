import web
import numpy
import sys
import cPickle
import json
import re

sys.path.append('../rnn')

import elmanNet as Net

word2index=cPickle.load(open('../word.corpus/50/word2Index.50.pkl','rb'))
model=Net.model()
model.loadRNNModelFromFile('../word.corpus/model')



urls=('/','index','/api','api')

app=web.application(urls,globals())

pattern = re.compile(ur'[\u4e00-\u9fa5]+')


class api:
    def GET(self):
        data=web.input()
        key=data.key

        res_key = [ word2index.get(x.encode('utf-8'),0) for x in key]

        p_cm_key=model.p_cm(res_key)

        p_key=p_cm_key[1]/(p_cm_key[0]+p_cm_key[1])

        return float('%0.5f'%p_key)

class index:

    def nGram(self,token,words):

        l=len(token)
        if l>1:
            for i in [2,3,4]:
                for j in range(l-i+1):
                    word=token[j:j+i]
                    words.add(word)
                    # print word

    def PCM(self,word):
    	res = [ word2index.get(x.encode('utf-8'),0) for x in word]
    	p_cm=model.p_cm(res)
    	p=p_cm[1]/(p_cm[0]+p_cm[1])
    	return float('%0.5f'%p)

    def GET(self):
    	web.seeother('/static/index.html')

    def POST(self):
    	data=web.data()
    	dataJson=json.loads(data)

    	sentence=dataJson['sentence']

    	#chinese only
    	words=set()
    	token=''
    	for x in sentence:
    		if pattern.match(x):
    			token+=x
    		else:
    			self.nGram(token,words)
    			token=''
    	self.nGram(token,words)
    	words=list(words)


    	res={}
    	for word in words:
    		res[word]=self.PCM(word)

    	res=sorted(res.iteritems(),key = lambda d:d[1],reverse=True)

    	res=json.dumps(res,ensure_ascii=False,indent=2)

    	return res

if __name__ == "__main__":
    app.run()