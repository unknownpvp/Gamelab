from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import math
import pickle
import operator
import json


class ImageSearch:

    def __inti__(self):
        self.index = {}
        self.queryvector = {}
        self.searchedquery = {}
        self.doctfidf = {}
        self.doctf = {}
        self.docidf = {}
        tokenizer = self.tokenizer
        self.finalimages = dict()

    def imageLoad(self, documents):
        docs = []
        cocourl = dict()

        for item in documents:
            docs.append(item[0])

        with open("image/panoptic_val2017.json", "r") as file:
            data = json.load(file)
            for item in data["images"]:
                if item["file_name"] in docs:
                    cocourl[item["file_name"]] = item["coco_url"]

        return cocourl

    def imageFile(self):
        file = open("image/Index.pkl", "rb")
        Index = pickle.load(file)
        self.index = Index

        file1 = open("image/tfidf.pkl", "rb")
        tfidf = pickle.load(file1)
        self.doctfidf = tfidf

        file2 = open("image/idf.pkl", "rb")
        idf = pickle.load(file2)
        self.docidf = idf

        file3 = open("image/tf.pkl", "rb")
        tf = pickle.load(file3)
        self.doctf = tf

        file4 = open("image/docsize.txt", "r")
        self.docsize = file4.readline()

        file5 = open("image/finalimages.pkl", "r")
        self.output_5k = file5

    def imagequery(self, terms):
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

    def imageresult(self):
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
        ans = self.imageLoad(data[0:20])
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
        return ans, data[0:20], tfidf

    def calMagnitude(self, vecValues):
        sum = 0
        for item in vecValues:
            sum = sum + (item * item)
        magnitude = math.sqrt(sum)
        return magnitude

    def calCosineSim(self):

        queryVecValues = list(self.queryvector.values())
        qvMag = self.calMagnitude(queryVecValues)
        file = open("image/tfidf.pkl", "rb")
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
        ans = self.imageLoad(data[0:20])
        return ans

    def dp(self, intersection, docVector):
        if len(intersection) != len(docVector):
            return 0
        return sum([x * y for x, y in zip(intersection, docVector)])
