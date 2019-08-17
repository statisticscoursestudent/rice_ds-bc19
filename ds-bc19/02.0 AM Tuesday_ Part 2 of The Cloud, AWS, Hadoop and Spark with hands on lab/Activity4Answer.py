
#####################################################################################################################
#
# kNN classifier for the 20 newsgroups data set, using cosine distance over bag-of-words count vectors
#
# run this code, and then type, for example: getPrediction ("god jesus allah", 30)
#
# This will come back with a prediction as to the membership of this text string in one of the 20 different
# nesgroups.  This particular query will return:
#
# [('/soc.religion.christian/', 15)]
#
# meaning that 15/30 closest articles to the string "god jesus allah" were in the '/soc.religion.christian/' newsgroup
# and that this was the most common newsgroup in the top 30.  Pretty good!
#
# But it is not always so good.  getPrediction ("how many goals Vancouver score last year?",30) returns:
#
# [('/comp.graphics/', 6)]
#
#####################################################################################################################

import re
import numpy as np

# load up all of the 19997 documents in the corpus
corpus = sc.textFile ("s3://chrisjermainebucket/comp330_A6/20_news_same_line.txt")

# each entry in validLines will be a line from the text file
validLines = corpus.filter(lambda x : 'id' in x)

# now we transform it into a bunch of (docID, text) pairs
keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:]))

# now we split the text in each (docID, text) pair into a list of words
# after this, we have a data set with (docID, ["word1", "word2", "word3", ...])
# we have a bit of fancy regular expression stuff here to make sure that we do not
# die on some of the documents
regex = re.compile('[^a-zA-Z]')
keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

# now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))

# now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
allCounts = allWords.reduceByKey (lambda a, b: a + b)

# and get the top 20,000 words in a local array
topWords = allCounts.top (20000, lambda x : x[1])

# and we'll create a RDD that has a bunch of (word, dictNum) pairs
# start by creating an RDD that has the number 0 thru 20000
# 20000 is the number of words that will be in our dictionary
twentyK = sc.parallelize(range(20000))

# now, we transform (0), (1), (2), ... to ("mostcommonword", 1) ("nextmostcommon", 2), ...
# the number will be the spot in the dictionary used to tell us where the word is located
dictionary = twentyK.map (lambda x : (topWords[x][0], x))

# next, we get a RDD that has, for each (docID, ["word1", "word2", "word3", ...]),
# ("word1", docID), ("word2", docId), ...
allWords = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

# and now join/link them, to get a bunch of ("word1", (dictionaryPos, docID)) pairs
allDictionaryWords = dictionary.join (allWords)

# and drop the actual word itself to get a bunch of (docID, dictionaryPos) pairs
justDocAndPos = allDictionaryWords.map (lambda x: (x[1][1], x[1][0]))

# now get a bunch of (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
allDictionaryWordsInEachDoc = justDocAndPos.groupByKey ()

# now, extract the newsgrouID, so that on input we have a bunch of
# (docID, [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs, but on output we 
# have a bunch of ((docID, newsgroupID) [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
# The newsgroupID is the name of the newsgroup extracted from the docID... for example 
# if the docID is "20_newsgroups/comp.graphics/37261" then the newsgroupID will be "s/comp.graphics/"
regex = re.compile('/.*?/')
allDictionaryWordsInEachDocWithNewsgroup = allDictionaryWordsInEachDoc.map (lambda x: ((x[0], regex.search(x[0]).group (0)), x[1]))

# this function gets a list of dictionaryPos values, and then creates a bag-of-words array
# corresponding to those values... for example, if we get [3, 4, 1, 1, 2] we would in the
# end have [0, 2, 1, 1, 1, ...] because 0 appears zero times, 1 appears twice, 2 appears once, etc.
def buildArray (listOfIndices):
        returnVal = np.zeros (20000)
        for index in listOfIndices:
                returnVal[index] = returnVal[index] + 1
        return returnVal

# this gets us a bunch of ((docID, newsgroupID) [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
# and converts the dictionary positiions to a bag-of-word numpy array
allDocsAsNumpyArrays = allDictionaryWordsInEachDocWithNewsgroup.map (lambda x: (x[0], buildArray (x[1])))

# print a few of the docs
allDocsAsNumpyArrays.top (10)

#####################################################################################################################


