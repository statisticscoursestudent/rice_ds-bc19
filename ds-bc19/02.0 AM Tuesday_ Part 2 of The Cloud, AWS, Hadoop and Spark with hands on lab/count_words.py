
def countWords (fileName):
	textfile = sc.textFile(fileName)
	lines = textfile.flatMap(lambda line: line.split(" "))
	counts = lines.map (lambda word: (word, 1))
	aggregatedCounts = counts.reduceByKey (lambda a, b: a + b)
	return aggregatedCounts.top (200, key=lambda p : p[1])


def countWords2 (fileName):
	textfile = sc.textFile(fileName)
	lines = textfile.flatMap(lambda line: line.split(" "))
	lines = lines.filter (lambda word: True if len (word) > 1 else False)
	counts = lines.map (lambda word: (word.lower (), 1))
	aggregatedCounts = counts.reduceByKey (lambda a, b: a + b)
	return aggregatedCounts.top (200, key=lambda p : p[1])

countWords ("s3://chrisjermainebucket/text/Holmes.txt")

countWords ("s3://chrisjermainebucket/text/")

countWords2 ("s3://chrisjermainebucket/text/Holmes.txt")

countWords2 ("s3://chrisjermainebucket/text/")
