
#####################################################################################################################
#
# kNN classifier for the 20 newsgroups data set, using TF-IDF
#
# run this code, and then type, for example: getPrediction ("god jesus allah", 30)
#
# This will come back with a prediction as to the membership of this text string in one of the 20 different
# nesgroups.  This particular query will return:
#
# [('/alt.atheism/', 12)]
#
# getPrediction ("how many goals Vancouver score last year?",30) returns:
#
# [('/rec.sport.hockey/', 23)]
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

# this function gets a list of dictionaryPos values, and then creates a TF vector
# corresponding to those values... for example, if we get [3, 4, 1, 1, 2] we would in the
# end have [0, 2/5, 1/5, 1/5, 1/5] because 0 appears zero times, 1 appears twice, 2 appears once, etc.
def buildArray (listOfIndices):
        returnVal = np.zeros (20000)
        for index in listOfIndices:
                returnVal[index] = returnVal[index] + 1
        mysum = np.sum (returnVal)
        returnVal = np.divide (returnVal, mysum)
        return returnVal

# this gets us a bunch of ((docID, newsgroupID) [dictionaryPos1, dictionaryPos2, dictionaryPos3...]) pairs
# and converts the dictionary positiions to a bag-of-words numpy array... 
allDocsAsNumpyArrays = allDictionaryWordsInEachDocWithNewsgroup.map (lambda x: (x[0], buildArray (x[1])))

# now, crete a version of allDocsAsNumpyArrays where, in the array, every entry is either zero or
# one.  A zero means that the word does not occur, and a one means that it does.
zeroOrOne = allDocsAsNumpyArrays.map (lambda x: (x[0], np.clip (np.multiply (x[1], 9e9), 0, 1)))

# now, add up all of those arrays into a single array, where the i^th entry tells us how many
# individual documents the i^th word in the dictionary appeared in
dfArray = zeroOrOne.reduce (lambda x1, x2: (("", np.add (x1[1], x2[1]))))[1]

# create an array of 20,000 entries, each entry with the value 19997.0
multiplier = np.full (20000, 19997.0)

# and get the version of dfArray where the i^th entry is the inverse-document frequency for the
# i^th word in the corpus
idfArray = np.log (np.divide (multiplier, dfArray))

# and finally, convert all of the tf vectors in allDocsAsNumpyArrays to tf * idf vectors
allDocsAsNumpyArrays = allDocsAsNumpyArrays.map (lambda x: (x[0], np.multiply (x[1], idfArray)))

# and finally, we have a function that returns the prediction for the label of a string, using a kNN algorithm
def getPrediction (textInput, k):
        #
        # push the text out into the cloud
        myDoc = sc.parallelize (('', textInput))
        #
        # gives us (word, 1) pair for each word in the doc
        wordsInThatDoc = myDoc.flatMap (lambda x : ((j, 1) for j in regex.sub(' ', x).lower().split()))
        #
        # this wil give us a bunch of (word, (dictionaryPos, 1)) pairs
        allDictionaryWordsInThatDoc = dictionary.join (wordsInThatDoc).map (lambda x: (x[1][1], x[1][0])).groupByKey ()
        #
        # and now, get tf array for the input string
        myArray = buildArray (allDictionaryWordsInThatDoc.top (1)[0][1])
        #
        # now, get the tf * idf array for the input string
        myArray = np.multiply (myArray, idfArray)
        #
        # now, we get the distance from the input text string to all database documents, using cosine similarity
        distances = allDocsAsNumpyArrays.map (lambda x : (x[0][1], np.dot (x[1], myArray)))
        #
        # get the top k distances
        topK = distances.top (k, lambda x : x[1])
        #
        # and transform the top k distances into a set of (newsgroupID, 1) pairs
        newsgroupsRepresented = sc.parallelize (topK).map (lambda x : (x[0], 1))
        #
        # now, for each newsgroupID, get the count of the number of times this newsgroup appeared in the top k
        numTimes = newsgroupsRepresented.aggregateByKey (0, lambda x1, x2: x1 + x2, lambda x1, x2: x1 + x2)
        #
        # and return the best!
        return numTimes.top (1, lambda x: x[1])

#####################################################################################################################


