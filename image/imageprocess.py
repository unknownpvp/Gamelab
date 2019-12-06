import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import math
import pickle


def createDoc(size):
    f = open(r'docsize.txt', "w")
    f.write(str(size))
    f.close()


def createIndex(index):
    f = open("Index.pkl", "wb")
    pickle.dump(index, f)
    f.close()


def calculateTFIDF(Index, total):
    doctfidf = {}
    docidf = {}
    doctf = {}
    for term, posting in Index.items():
        df = len(posting)
        idf = math.log2((total / df))

        for item in posting:
            if item[0] in doctfidf.keys():
                doctfidf[item[0]][term] = item[1] * idf
                docidf[item[0]][term] = idf
                doctf[item[0]][term] = item[1]
            else:
                doctfidf[item[0]] = {term: item[1] * idf}
                docidf[item[0]] = {term: idf}
                doctf[item[0]] = {term: item[1]}

    file1 = open('tfidf.pkl', "wb")
    pickle.dump(doctfidf, file1)
    file1.close()
    file2 = open('idf.pkl', "wb")
    pickle.dump(docidf, file2)
    file2.close()
    file3 = open('tf.pkl', "wb")
    pickle.dump(doctf, file3)
    file3.close()


imageset = open("finalimages.pkl","rb")
tempDict = pickle.load(imageset)

newDict = {}
tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
for key, value in tempDict.items():
    newDict[key] = list(tokenizer.tokenize(str(value).lower()))

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
for key, value in newDict.items():
    temp = []
    for word in value:
        if word not in stop_words:
            temp.append(ps.stem(word))
    newDict[key] = temp

docIndex = {}
for id, words in newDict.items():
    docIndex[id] = []
    for word in words:
        docIndex[id].append({word: [id, words.count(word)]})

Index = {}
for indexList in docIndex.values():
    for element in indexList:
        term = list(element.keys())
        value = list(element.values())
        if term[0] in Index:
            if value[0] not in Index[term[0]]:
                Index[term[0]].append(value[0])
        else:
            Index[term[0]] = [value[0]]

createIndex(Index)
calculateTFIDF(Index, len(docIndex))
createDoc(len(docIndex))
