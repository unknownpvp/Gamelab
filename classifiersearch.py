import csv
import math
import random
import math
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import defaultdict


def count_document(document, c):
    document_in_c = 0
    for doc in document:
        if c in doc:
            document_in_c += 1
    return document_in_c


def concatenate_text(categories, document, c):
    text_in_c = []
    for i in range(len(document)):
        if c in categories[i]:
            text_in_c.extend(document[i])
    return text_in_c


class naivebayes:
    dataset = pd.read_csv('metacritic_game_info.csv')
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    stops = stopwords.words('english')
    stemmer = PorterStemmer()
    total_words = []
    final_document = []
    weight_vectors = []
    posting_lists = {}
    vocabulary = []
    categories = []
    prior = {}
    condprob = defaultdict(dict)
    counts = []

    def __init__(self):
        dataset = self.dataset
        tokenizer = self.tokenizer
        stops = self.stops
        stemmer = self.stemmer
        final_document = self.final_document
        weight_vectors = self.weight_vectors
        posting_lists = self.posting_lists
        vocabulary = self.vocabulary
        categories = self.categories
        prior = self.prior
        condprob = self.condprob
        counts = self.counts

        for i in range(500):
            tokens = tokenizer.tokenize(dataset['Title'][i])
            tokens += tokenizer.tokenize(dataset['Genre'][i])
            final_tokens = []
            for token in tokens:
                token = token.lower()
                if token not in stops:
                    token = stemmer.stem(token)
                    final_tokens.append(token)
                    if token not in vocabulary:
                        vocabulary.append(token)
            final_document.append(final_tokens)

        for document in final_document:
            weight_vector = {}
            for term in document:
                if term not in weight_vector:
                    tf = document.count(term) / len(document)
                    df = sum(1 for document in final_document if term in document)
                    n = len(final_document)
                    weight = tf * math.log(n / df)
                    weight_vector[term] = weight

            weight_vectors.append(weight_vector)

        for i in range(len(weight_vectors)):
            document = weight_vectors[i]
            for token in document:
                if token not in posting_lists:
                    posting_lists[token] = []
                posting_lists[token].append([i, document[token]])
                posting_lists[token] = sorted(
                    posting_lists[token], key=lambda x: x[1], reverse=True)

        total_document = len(final_document)
        total_term = len(vocabulary)
        genres = dataset['Genre']

        for genre in genres:
            if genre not in categories:
                categories.append(genre)

        for i in categories:
            document_in_c = count_document(genres, i)
            prior[i] = document_in_c / float(total_document)
            text_in_c = concatenate_text(genres, final_document, i)
            for term in vocabulary:
                Tct = text_in_c.count(term)
                counts.append(Tct)
                condprob[term][i] = (Tct + 0.001) / (len(text_in_c) + total_term)

    def classify(self, query):
        genre = []
        terms = self.tokenizer.tokenize(query)
        for term in terms:
            term = term.lower()
            if term not in self.stops:
                term = self.stemmer.stem(term)
                genre.append(term)

        score = {}
        for i in self.categories:
            score[i] = self.prior[i]
            for term in genre:
                if term in self.condprob:
                    score[i] *= self.condprob[term][i]

        total = sum(score.values())
        ans = {}
        for i in sorted(score, key=score.get, reverse=True):
            ans[i] = score[i] / float(total)

        return ans