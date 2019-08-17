# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

# this returns a number whose probability of occurence is p
def sampleValue (p):
        return np.flatnonzero (np.random.multinomial (1, p, 1))[0]

# there are 2000 words in the corpus
alpha = np.full (2000, .1)

# there are 100 topics
beta = np.full (100, .1)

# this gets us the probabilty of each word happening in each of the 100 topics
wordsInTopic = np.random.dirichlet (alpha, 100)

# wordsInCorpus[i] will give us the number of each word in the document
wordsInCorpus = {}

# generate each doc
for doc in range (0, 50):
        #

        # no words in this doc yet
        wordsInDoc = {}
        #

        # get the topic probabilities for this doc
        topicsInDoc = np.random.dirichlet (beta)
        #

        # generate each of the 1000 words in this document
        for word in range (0, 1000):
                #

                # select the topic and the word
                whichTopic = sampleValue (topicsInDoc)
                whichWord = sampleValue (wordsInTopic[whichTopic])
                #

                # and record the word
                wordsInDoc [whichWord] = wordsInDoc.get (whichWord, 0) + 1
                #
        # now, remember this document
        wordsInCorpus [doc] = wordsInDoc
        