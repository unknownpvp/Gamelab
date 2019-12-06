from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
import pickle
import operator
import csv


class TextSearch:

    def __inti__(self):
        self.index = {}
        self.queryvector = {}
        self.searchedquery = {}
        self.doctfidf = {}
        self.doctf = {}
        self.docidf = {}
        tokenizer = self.tokenizer

    def dataLoad(self, documents):
        docs = []
        for item in documents:
            docs.append(item[0])

        Dict = {}
        with open('comments.csv', encoding="utf8") as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i = 1
            for row in reader:
                commentdata = row[0]
                if i in docs:
                    for term in self.searchedquery:
                        commentdata = commentdata.replace(term, "" + term + "")
                Dict[i] = commentdata
                i = i + 1
        return Dict

    def dataFile(self):
        file = open("Index.pkl", "rb")
        Index = pickle.load(file)
        self.index = Index

        file1 = open("tfidf.pkl", "rb")
        tfidf = pickle.load(file1)
        self.doctfidf = tfidf

        file2 = open("idf.pkl", "rb")
        idf = pickle.load(file2)
        self.docidf = idf

        file3 = open("tf.pkl", "rb")
        tf = pickle.load(file3)
        self.doctf = tf

        file4 = open('docsize.txt', "r")
        self.docsize = file4.readline()

    def searchquery(self, terms):
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()
        result = []
        for word in terms:
            if word not in stop_words:
                result.append(ps.stem(word))

        self.searchedquery = result

    def calculateTFIDF(self):
        query = []

        for term in self.searchedquery:
            if term in self.index.keys():
                df = len(self.index[term])
                query[term] = self.searchedquery.count(term) * math.log2(int(self.docsize) / df)
            else:
                continue
        self.queryvector = query

    def searchresult(self):
        docs = []

        for term, posting in self.index.items():
            if term in self.searchedquery:
                for item in posting:
                    docs.append(item[0])

        docs = list(set(docs))
        sim = {key: self.doctfidf[key] for key in docs}
        sum = dict.fromkeys(docs, 0)

        for index, terms in sim.items():
            for term in list(terms.keys()):
                if term in self.searchedquery:
                    sum[index] = sum[index] + terms[term]
                    continue
                else:
                    del terms[term]

        data = sorted(sum.items(), key=operator.itemgetter(1), reverse=True)
        ans = self.dataLoad(data[0:20])
        idf = ({key: self.docidf[key] for key in list(sum.keys())})
        tf = {key: self.doctf[key] for key in list(sum.keys())}

        for index, terms in idf.items():
            for term in list(terms.keys()):
                if term in self.searchedquery:
                    continue
                else:
                    del terms[term]

        for index, terms in tf.items():
            for term in list(terms.keys()):
                if term in self.searchedquery:
                    continue
                else:
                    del terms[term]

        tfidf = {key: self.doctfidf[key] for key in list(sum.keys())}
        return ans, idf, tfidf, data[0:20], tf

    def calMagnitude(self, vecValues):
        sum = 0
        for item in vecValues:
            sum = sum + (item * item)
        magnitude = math.sqrt(sum)
        return magnitude

    def calCosineSim(self):

        queryVecValues = list(self.queryvector.values())
        qvMag = self.calMagnitude(queryVecValues)
        file = open("tfidf.pkl", "rb")
        tfidf = pickle.load(file)
        file.close()
        self.doctfidf = tfidf

        mag = {}
        cosine = {}

        for index, weight in self.doctfidf.items():
            intersection = list(set(self.queryvector.keys()) & set(weight.keys()))

            mag[index] = self.calMagnitude(list(weight.values()))
            cosine[index] = (self.dp(intersection, weight)) / (mag[index] * qvMag)

        data = sorted(cosine.items(), key=operator.itemgetter(1), reverse=True)
        ans = self.dataLoad(data[0:20])
        return ans

    def dp(self, intersection, docVector):
        if len(intersection) != len(docVector):
            return 0
        return sum([x * y for x, y in zip(intersection, docVector)])
