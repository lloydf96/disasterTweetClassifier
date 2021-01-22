# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import nltk
import re
import tensorflow as tf
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Flatten, Dense,Embedding,LSTM,SimpleRNN,Concatenate,Input
import matplotlib.pyplot as plt
import keras.metrics as metrics
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


class nlpTest():
    
    def __init__(self):
        self.importTrainingData()
        
    
    def importTrainingData(self):
        self.maxLen = 25
        self.percent_train = 0.90
        self.validation_samples = 2000
        self.embeddingDim=32
        self.maxWords=10000
        self.maxHashTags=100
        self.maxUserTags=100
        self.maxLocations = 500
        self.epochs=10
        
        self.data=pd.read_csv("train.csv",delimiter=',')
        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)
        training_samples = int(self.percent_train*len(self.data.id))
        
        self.dataFit=self.data[:training_samples]
        self.dataTest=self.data[training_samples:]
        self.YFit=self.dataFit.loc[:,['target']]
        self.YTest=self.dataTest.loc[:,['target']]
        
    @staticmethod
    #cleans list of words
    def wordProcessing(wordList):
        stopword=stopwords.words('english')
        wordnet_lemmatizer = WordNetLemmatizer()
        stemmer=SnowballStemmer('english')
        replHash=re.compile(r'\s#\w+|^#\w+') #Compile Hash Pattern
        replAt=re.compile(r'\s@\w+|^@\w+')
        wordList=[word for word in wordList if word.isalpha() or (bool(replHash.search(word)) or bool(replAt.search(word)))]
        replPunct = re.compile(r'[^\w\s]')
        
        #lowercase the words
        wordList=[word.lower() for word in wordList]
        
        #see if words in stopword remove if yes
        #wordList=[word for word in wordList if word not in stopword]
        
        #lemmatize the words
        wordList=[ word if (bool(replHash.search(word)) or bool(replAt.search(word))) else replPunct.sub('',word) for word in wordList]
        
        wordList=[ word if (bool(replHash.search(word)) or bool(replAt.search(word))) else wordnet_lemmatizer.lemmatize(word) for word in wordList if word]
        #wordList=[ word if bool(replHash.search(word)) or bool(replAt.search(word)) else stemmer.stem(word) for word in wordList]
        
        wordTagged=pd.DataFrame(nltk.pos_tag(wordList),columns=['words','POS'])
        #print(wordTagged.loc[list(wordTagged['POS']=='CD'),'words'])
        #wordTagged.loc[list(wordTagged['POS']=='CD'),'words']='numtag'
        #wordTagged=wordTagged.replace(index=[i for i,val in wordTagged.POS.iteritems() if val in ['CD']])
        wordList=list(wordTagged.words)
        return wordList
    
    #cleans and convert dataset to list of tokens
    def text_to_tokens(self,data):
        #self.wordMaster=[]
        wordList = []
        textTokens=pd.DataFrame(columns={'text':object})
        
        replURL=re.compile('(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-?=%.]+')
        tk=TweetTokenizer()
        
        for i,row in data.loc[:,["id","text"]].iterrows():
            
            text=row.text
            text=replURL.sub('urltag',text)
            #token_list = nltk.word_tokenize(text)
            token_list = tk.tokenize(text)
            wordList.append(token_list)
            
        textTokens.text=[self.wordProcessing(text) for text in wordList]
        
        return textTokens
      
    #generates a list of words,hashtags,@words,locations and keywords
    def prepTokens(self,textTokens,data):
        wordList = []
        hashWord = []
        userWord = []
        
        replHash=re.compile(r'\s#\w+|^#\w+') #Compile Hash Pattern
        replAt=re.compile(r'\s@\w+|^@\w+')
        replSpace=re.compile(r' +')#Compile @ pattern
        wordList = textTokens.text
        
        wordList=[word for sublist in wordList for word in sublist]
        
        hashWord=[word for word in wordList if bool(replHash.search(word))]
        userWord=[word for word in wordList if bool(replAt.search(word))]
        #Counter to get max list of words
        Counter_WordList=Counter(wordList)
        wordMaster=pd.DataFrame(Counter_WordList.most_common(self.maxWords),columns=['words','freq'])
        
        if 'urlTag' not in list(wordMaster.words):
            wordMaster.append(['urlTag']) 
            
        if 'numTag' not in list(wordMaster.words):
            wordMaster.append(['numTag'])
            
        #Counter to get Max limit of hasttags   
        Counter_hashWord=Counter(hashWord)
        hashWord=Counter_hashWord.most_common(self.maxHashTags)
        hashTable=pd.DataFrame(hashWord,columns=['hashTag','Freq'])
        
        #Counter to get Max limit of userWords
        Counter_userWord=Counter(userWord)
        userWord=Counter_userWord.most_common(self.maxUserTags)
        userTable=pd.DataFrame(userWord,columns=['userName','Freq'])
        
        data.loc[:,'keyword'] = data.loc[:,'keyword'].replace(np.nan,'no_keyword')
        keywordTable=list(set([replSpace.sub('',str(word)).lower() for word in [word for i,word in data.keyword.iteritems()]]))
        data.loc[:,'location'] = data.loc[:,'location'].replace(np.nan,'no_location')
        locationWords=[replSpace.sub('',str(word)).lower() for word in [word for i,word in data.location.iteritems()]]
        c_location=Counter(locationWords)
        location=pd.DataFrame(data=c_location.most_common(self.maxLocations),columns=['location','freq'])
        
        #self.wordPOS=nltk.pos_tag(self.wordMaster)
        return wordMaster,hashTable,userTable,keywordTable,location
        #print (self.hashTable.loc[:,'hashTag'])
        #print (self.userTable.loc[:,'userName'])
        #print (self.wordMaster)
    
    #generates word dictionary of type{word:index} using list of tokens
    def generate_word_dict(self,data,textTokens):
        
        #generates different categories of words
        wordMaster,hashTable,userTable,keywordTable,location=self.prepTokens(textTokens,data)
        #wordMaster,hashTable,userTable=self.prepTokens(textTokens,data)
        wordList=list(set(list(wordMaster.words)+list(hashTable.hashTag)+list(userTable.userName)\
                          +['newuser','newhash','newlocation','newkeyword']))
        wordDict=dict(zip(wordList,np.arange(1,len(wordList)+1)))
        return wordDict
     
    #converts text entries to list of integers
    def text_to_sequence(self,data,wordDict,textTokens):
      
         sequencesWord=pad_sequences([[self.sequencingFunction(word,wordDict) for word in text] for text in textTokens.text],maxlen=self.maxLen)
         
         sequencesLocation=[wordDict[word] if word in wordDict else \
                                           wordDict['newlocation'] for word in data.location]

         sequencesKeyword=[wordDict[word] if word in wordDict else \
                                          wordDict['newkeyword'] for word in data.keyword]
         
         XIntSeq1=pd.DataFrame(data=sequencesWord)
         
         XIntSeq2=pd.DataFrame(data=sequencesLocation,columns=[self.maxLen+1]).join (pd.DataFrame(data=sequencesKeyword,columns=[self.maxLen+2]))
             
         return XIntSeq1,XIntSeq2
       
    def sequencingFunction(self,word,wordDict):
        replHash=re.compile(r'\s#\w+|^#\w+') #Compile Hash Pattern
        replAt=re.compile(r'\s@\w+|^@\w+') #Compile @ pattern
        if word in wordDict:
            return wordDict[word]
        elif bool(replHash.search(word)):
            return wordDict['newhash']
        elif bool(replAt.search(word)):
            return wordDict['newuser']
        else:
            return 0
      
    def DLTrainPreprocessing(self,data):
        self.importTrainingData()
        textTokensFit = self.text_to_tokens(self.dataFit)
        self.wordDict = self.generate_word_dict(self.dataFit,textTokensFit)
        self.XFit1,self.XFit2 = self.text_to_sequence(self.dataFit,self.wordDict,textTokensFit)
        
        
        textTokensTest = self.text_to_tokens(self.dataTest)
        self.XTest1,self.XTest2 = self.text_to_sequence(self.dataTest,self.wordDict,textTokensTest)
        
    def DLTraining(self):
        
        dictLength = len(self.wordDict)
        
        maxLen = self.maxLen
        
        self.model = self.nlpLSTMModel(dictLength+1,self.embeddingDim,maxLen,self.XFit1,self.XFit2,self.YFit,self.XTest1,self.XTest2,self.YTest)
    
    def DLPredict(self):
        
        self.dataPredict=pd.read_csv("test.csv",delimiter=',')
        textTokensPredict = self.text_to_tokens(self.dataPredict)
        self.XPredict = self.text_to_sequence(self.dataPredict,self.wordDict,textTokensPredict)
        YPredict = self.model.predict(self.XPredict)
        Submission = pd.DataFrame(data=self.dataPredict.loc[:,'id']).join(pd.DataFrame(data=[1 if y>0.5 else 0 for y in YPredict],columns = ['target']))
        Submission.to_csv('Submission.csv',index = False)
    
    def nlpSimpleRNNModel(self,dictLength,embeddingDim,maxLen,XFit,YFit,XTest,YTest):
        model = Sequential()
        model.add(Embedding(dictLength,embeddingDim, input_length=maxLen))
        model.add(SimpleRNN(32, activation='relu',return_sequences = True))
        model.add(SimpleRNN(32, activation='relu',return_sequences = True))
        model.add(SimpleRNN(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['acc'])
        
        history = model.fit(XFit, YFit,
                            epochs=self.epochs,
                            batch_size=128,
                            validation_data=(XTest, YTest))
        
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        return model
        
    def nlpLSTMModel(self,dictLength,embeddingDim,maxLen,XFit1,XFit2,YFit,XTest1,XTest2,YTest):
        
        input1 = Input(shape=(maxLen,))
        input2 = Input(shape=(2,))
        embedding1 = Embedding(dictLength,embeddingDim, input_length=maxLen)(input1)
        
        embedding2 = Embedding(dictLength,embeddingDim, input_length=2)(input2)
        layer1 = LSTM(16,activation='relu')(embedding1)
        layer2 = Flatten()(embedding2)
        layer2 = Dense(16, activation='relu')(layer2)
        
        merge = Concatenate()([layer1,layer2])
        layer3 = Dense(8,activation = 'relu')(merge)
        layer4=Dense(1, activation='sigmoid')(layer3)
        
        
        model=Model(inputs = [input1,input2],outputs = layer4)
        
        model.summary()
        
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['acc'])
        
        history = model.fit([XFit1,XFit2], YFit,
                            epochs=self.epochs,
                            batch_size=128,
                            validation_split=0.2)
        
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        return model
     
    def nlpNNModel(self,dictLength,embeddingDim,maxLen,XFit,YFit,XTest,YTest):
        model = Sequential()
        model.add(Embedding(dictLength,embeddingDim, input_length=maxLen))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['acc'])
        
        history = model.fit(XFit, YFit,
                            epochs=self.epochs,
                            batch_size=32,
                            validation_data=(XTest, YTest))
        
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        return model
        
        
  
        