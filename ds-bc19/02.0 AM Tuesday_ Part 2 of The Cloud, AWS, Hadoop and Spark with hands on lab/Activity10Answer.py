

#####################################################################################################################
#
# Linear regression classifier for the 20 newsgroups data set.  Checks whether the given input string is
# about religion or not.
#
# run this code, and then type, for example: getPrediction ("god jesus allah")
#
# This will come back with a prediction as to the membership of this text string in one of the 20 different
# nesgroups.  This particular query will return:
#
# 'about religion'
#
# getPrediction ("how many goals Vancouver score last year?") returns:
#
# 'not about religion'
#
# getPrediction ("I am not totally sure that I believe in the afterlife") returns:
#
# 'about religion'
#
# getPrediction ("I am not totally sure that I believe in swimming soon after I finish eating") returns:
#
# 'not about religion'
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

# create a 20,000 by 1000 matrix where each entry is sampled from a Normal (0, 1) distribution...
# this will serve to map our 20,000 dimensional vectors down to 1000 dimensions
mappingMatrix = np.random.randn (20000, 1000)

# now, map all of our tf * idf vectors down to 1000 dimensions, using a matrix multiply...
# this will give us an RDD consisteing of ((docID, newsgroupID), numpyArray) pairs, where
# the array is a tf * idf vector mapped down into a lower-dimensional space
allDocsAsLowerDimNumpyArrays = allDocsAsNumpyArrays.map (lambda x: (x[0], np.dot (x[1], mappingMatrix)))

# and now take an outer product of each of those 1000 dimensional vectors with themselves
allOuters = allDocsAsLowerDimNumpyArrays.map (lambda x: (x[0], np.outer (x[1], x[1])))

# and aggregate all of those 1000 * 1000 matrices into a single matrix, by adding them all up...
# this will give us the complete gram matrix
gramMatrix = allOuters.aggregate (np.zeros ((1000, 1000)), lambda x1, x2: x1 + x2[1], lambda x1, x2: x1 + x2)

# take the inverse of the gram matrix
invGram = np.linalg.inv (gramMatrix)

# now, go through allDocsAsNumpyArrays and multiply each of those 1000-dimensional vectors by the
# gram matrix... allRows will have a bunch of ((docID, newsgroupID), numpyArray) pairs, where the
# array is the mapped TF * IDF vector, multiplied by the Gram matrix
allRows = allDocsAsLowerDimNumpyArrays.map (lambda x: (x[0], np.dot (invGram, x[1])))

# and now, multiply each entry allRows by 1 if the document came from a religion-oriented newsgroup;
# that is, '/soc.religion.christian/' or '/alt.atheism/' or '/talk.religion.misc/', and by a -1 if
# it did not come from a religion-oriented newsgroup
allRowsMapped =  allRows.map (lambda x: (x[0], x[1] if (x[0][1] == '/soc.religion.christian/' or x[0][1] == '/alt.atheism/' or x[0][1] == '/talk.religion.misc/') else np.multiply (-1, x[1])))

# finally, we can compute the vector of regression parameters by simply adding up all of the vectors
# that we had in the allRowsMapped RDD
regressionParams = allRowsMapped.aggregate (np.zeros (1000), lambda x1, x2: x1 + x2[1], lambda x1, x2: x1 + x2)

regressionParams
