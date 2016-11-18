# -*- coding: utf-8 -*-
"""
Created on Mon Nov 07 18:57:55 2016

@author: somrup
"""
import nltk
import os
import csv
import string
from nltk.parse.stanford import StanfordDependencyParser
from collections import Counter
from nltk.tag.stanford import StanfordPOSTagger
from nltk.stem import PorterStemmer
import itertools
from nltk import word_tokenize
from nltk.parse import stanford
# In[3]:



# In[7]:
from nltk.stem import PorterStemmer

import csv
import pandas as pd
data=pd.read_csv('NLP Technology.csv')
jarpath = "C:\Users\somrup\Documents\Algo 1\nlp\stanford-corenlp-full-2015-12-09"
#os.environ['STANFORD_CORENLP'] = jarpath
#os.environ['STANFORD_PARSER'] = 'C:\Users\somrup\Documents\Algo 1\nlp\stanford-corenlp-full-2015-12-09\stanford-parser-full-2015-12-09\stanford-parser-full-2015-12-09\stanford-parser.jar'
#os.environ['CLASSPATH'] ='/Users/somrup/Documents/Algo 1/nlp/stanford-corenlp-full-2015-12-09/stanford-parser-full-2015-12-09/stanford-parser.jar'
file=open("lifestyle.txt", "w")


stanford_modelpath='/Users/somrup/Documents/Algo 1/nlp/stanford-corenlp-full-2015-12-09/stanford-parser-full-2015-12-09/stanford-parser-full-2015-12-09'
eng_modelpath='/Users/somrup/Documents/Algo 1/nlp/models/edu/stanford/nlp/models/lexparser/englishRNN.ser.gz'
jar_path=stanford_modelpath+"/stanford-parser.jar"
mymodel_path=stanford_modelpath+"/stanford-parser-3.6.0-models.jar"
parser=StanfordDependencyParser(model_path=eng_modelpath,path_to_models_jar=mymodel_path,path_to_jar=jar_path)

jar='/Users/somrup/Documents/Algo 1/nlp/stanford-postagger-full-2015-12-09/stanford-postagger-full-2015-12-09/stanford-postagger.jar'
model='/Users/somrup/Documents/Algo 1/nlp/stanford-postagger-full-2015-12-09/stanford-postagger-full-2015-12-09/models/english-left3words-distsim.tagger'
tagger = StanfordPOSTagger(model, jar)

file_strong_adj='/Users/somrup/Documents/Algo 1/nlp/avinava/strong.txt'
file_stop_words='/Users/somrup/Documents/Algo 1/nlp/avinava/stopwords.txt'
file_weak_adj='/Users/somrup/Documents/Algo 1/nlp/avinava/weak.txt'#to define the full path where the weak adjective list is kept

for row in range(4,100):
   text=data['Lifestyle'][row]
   text1=text.lower()   
   no_punctuation = text1.translate(None, string.punctuation)
   tokens=nltk.word_tokenize(no_punctuation)
   postext=tagger.tag(tokens)
   
   stop_words_lines = [line.rstrip('\r\n') for line in open(file_stop_words)]
   final_stop_words_list=set(stop_words_lines)
   filtered_sent_words=[]
   for wrd in tokens:
       if wrd not in final_stop_words_list:
           filtered_sent_words.append(wrd)
   num_strongadj=0
   num_weakadj=0
   strong_adj_lines = [line.rstrip('\r\n') for line in open(file_strong_adj)]
   final_strong_adj_list=set(strong_adj_lines)
   for str_adj in final_strong_adj_list:
       for wrd_sent in filtered_sent_words:
           if str_adj==wrd_sent:
               num_strongadj=num_strongadj+1
    
   
   weak_adj_lines = [line.rstrip('\r\n') for line in open(file_weak_adj)]
   final_weak_adj_list=set(weak_adj_lines)
   for str_adj in final_weak_adj_list:
       for wrd_sent in filtered_sent_words:
           if str_adj==wrd_sent:
               num_weakadj=num_weakadj+1
   model1='/Users/somrup/Documents/Algo 1/nlp/stanford-postagger-full-2015-12-09/stanford-postagger-full-2015-12-09/models/english-bidirectional-distsim.tagger'
   st=StanfordPOSTagger(model1,jar)
   postags_list=st .tag(no_punctuation.split())
   num_pronoun=0
#to find out if pronoun is present or not
   for word,pos in postags_list:
       if pos in ['PRP','PRP$','WP','WP$']:
           num_pronoun=num_pronoun+1
   postags_list_sent=[]
   filtered_sent_tagf=[]
   ps=PorterStemmer()
   for i in filtered_sent_words:    
       postags_list_sent.append(st.tag(i.split()))
      
            #to stem the words from the filtered sentences i.e after removal of stopwords
            # stemming done only if not a noun and pronoun
   for i_n in postags_list_sent:
    for word,pos in i_n:
        if pos in ['NN','NNS','NNP','NNPS','PR','PRP$','WP','WP$']:
            filtered_sent_tagf.append(word)
        else:
            filtered_sent_tagf.append(ps.stem(word))
   result=parser.raw_parse(text)
   dep=result.next()
   k_tri=list(dep.triples())
   count_total=Counter(tags for word1,tags,word2 in k_tri)
   num_acomp=num_ccomp=num_amod=num_advcl=num_advmod=num_xcomp=num_nummod=num_dep=num_dobj=num_iobj=0
   for w1,tag,w2 in k_tri:
        if tag=='dobj':
            num_dobj=num_dobj+1
        if w1[1] in ['VB','VBN','VBD','VBG','VBP','VBZ']:
            num_root=1
        if tag=='dep':
            num_dep=num_dep+1
        if tag=='iobj':
            num_iobj=num_iobj+1
        if w1[1] in ['VB','VBN','VBD','VBG','VBP','VBZ']:
            num_root=1
        if tag=='nummod':
            num_nummod=num_nummod+1
    
   list_new=list(count_total.iterkeys())
#print list_new
#print count_total[list_new[0]]
   for i in range(len(list_new)):#opinion
        if list_new[i]=='acomp':
            num_acomp=count_total[list_new[i]]
        if list_new[i]=='ccomp':
            num_ccomp=count_total[list_new[i]]
        if list_new[i]=='advmod':
            num_advmod=count_total[list_new[i]]
        if list_new[i]=='advcl':
            num_advcl=count_total[list_new[i]]
        if list_new[i]=='amod':
            num_amod=count_total[list_new[i]]
        if list_new[i]=='xcomp':
            num_xcomp=count_total[list_new[i]]
        if list_new[i]=='nummod' or list_new[i]=='num':
            num_nummod=1#count_total[list_new[i]]
   strfile= str(num_strongadj)+','+str(num_weakadj)+','+str(num_acomp)+','+str(num_ccomp)+','+ str(num_advmod)+','+ str(num_advcl)+','+ str(num_amod)+','+str(num_xcomp)+','+str(num_nummod)+','+str(num_dobj)+','+str(num_iobj)+','+str(num_dep)+','+str(num_root)+','+data['Class'][row]
   file.write(strfile)
   strfile=""
file.close()

       
    

