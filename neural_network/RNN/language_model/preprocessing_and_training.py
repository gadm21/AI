
import os
os.chdir("neural_network/RNN/language_model/")
import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime
from utils import softmax, save_model_parameters_theano, load_model_parameters_theano

import matplotlib.pyplot as plt


'''

the data is taken from a 15,000 reddit comments

Before training our RNN, we need to process the data following these steps
1- tokenizing words
    a- similar to splitting by space 
    b- we tokenize statements from text, then words from sentences
2- removing infrequent words
    a- Part of the reason is to reduce the training data to reduce training time
    b- The important reason is that we don't have many contextual information of infrequent words,
    and the model needs to see each word in different contexts to learn how it is used
    c- we replace infrequent words with a constant token (like "UNKOWN_TOKEN"). When the model starts
    generating text, it will generate the constant token that we added like any other word, and we can
    choose what we do with it. either replacing it again in the generated text with a word of our choice
    or totally discard generated sentences that contain this constant token.
3- prepend special start and end tokens
    a- to make the model generate a full sentence, we add a special starting and ending token to the 
    training data so that at testing time we will start by giving the model the starting token and the 
    model will start by generating the next word, which is the actual starting word.
4- build training data matricies
    a- the input to the RNN is not strings, it's a vector. So we'll make a mapping on our training data,
    word-to-index and index-to-word. Thus, the input to the RNN model is of the form [0, 123, 564, 22] 
    where [0] is the starting token, and the y-label is [123, 564, 22]. Notice the relation between the
    data and its label (shifting).
'''

vocabulary_size= 5000
data_folder= 'data/'
unkown_token= 'UNKOWN_TOKEN'
start_token= 'START_TOKEN'
end_token= 'END_TOKEN'



with open(data_folder+ "reddit_comments.csv", encoding= 'utf-8') as f:
    reader= csv.reader(f, skipinitialspace= True)
    next(reader)
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    sentences = ["%s %s %s" % (start_token, x, end_token) for x in sentences]

tokenized_sentences= [nltk.word_tokenize(sentence) for sentence in sentences]
