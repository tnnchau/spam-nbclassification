# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 13:58:13 2020

@author: ASUS
"""
import pandas as pd
import nltk 
from nltk.tokenize import RegexpTokenizer

from sklearn.model_selection import train_test_split
from collections import Counter

data = pd.read_csv('spam.csv',encoding='latin-1')
print(data.columns)
print(data.head(5))
data = data[['v1','v2']]
print(data.head(5))

data_train, data_test = train_test_split(data,test_size=0.2,random_state=4)
print("Total data: ", len(data))
print("Train data: ", len(data_train))
print("Test data: ", len(data_test))

labels = list(data_train['v1'])
messages = list(data_train['v2'])
print("Total labels: ", len(labels))
print("Total messages: ", len(messages))
print(labels[:5])
print(messages[:5])

#count word
word_set = set()
for msg in messages:
    tokens = nltk.word_tokenize(msg)
    for tk in tokens:
        word_set.add(tk)
    V = len(word_set)
print("Total word: " ,V)
    
#count label
counts_label = Counter(labels)
print(counts_label)
N = counts_label['spam'] + counts_label['ham']
print('p(spam)=', counts_label['spam']/N)
print('p(ham)=',counts_label['ham']/N)

#count pair: label word
label_word_list = []
for i in range(len(labels)):
    token = nltk.word_tokenize(messages[i])
    for tk in token:
        label_word_list.append(labels[i]+'_'+tk)
counts_label_word = Counter(label_word_list)
print("Total word with label: ",len(counts_label_word))
i=0
for k in counts_label_word.keys():
    print(k,':',counts_label_word[k])
    i+=1
    if i>10:
        break
#count counts_label according to word
label_dict =  {'spam':0,'ham':0}
for i in range(len(labels)):
    tokens = nltk.word_tokenize(messages[i])
    label_dict[labels[i]] += len(tokens)
print(label_dict)
x = (counts_label_word['spam_a']+1)/(label_dict['spam']+V)
print(counts_label_word['spam_a'])
print(x)

#tagging
text = input("Text to tag:")
tokenizer = RegexpTokenizer('\w+')
temp = tokenizer.tokenize(text)
pspam = counts_label['spam']/N
pham = counts_label['ham']/N
for i in temp:
    temp_pspam = (counts_label_word['spam_'+i]+1)/(label_dict['spam'] + V)
    temp_pham = (counts_label_word['ham_'+i]+1)/(label_dict['ham'] + V)
    pspam *= temp_pspam
    pham *= temp_pham
if (pham > pspam):
    print("Tag: ham")
else:
    print("Tag: spam")
