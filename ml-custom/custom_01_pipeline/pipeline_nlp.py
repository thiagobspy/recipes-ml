import re

from nltk import word_tokenize
from pandas import read_csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

def clean_phrases(phrases, func=str):
    phrases_corpus = []
    for phrase in phrases:
        phrase = re.sub(r"'s", " is ", phrase)
        phrase = re.sub(r"'ve", " have ", phrase)
        phrase = re.sub(r"n't", " not ", phrase)
        phrase = re.sub(r"'re", " are ", phrase)
        phrase = re.sub(r"'d", " would ", phrase)
        phrase = re.sub(r"'ll", " will ", phrase)
        phrase = re.sub(r",", "", phrase)
        phrase = re.sub(r"\.", "", phrase)
        phrase = re.sub(r"\.  \.  \.", "", phrase)
        phrase = re.sub(r"!", "", phrase)
        phrase = re.sub(r"\/", "", phrase)
        phrase = re.sub(r"-", "", phrase)
        phrase = re.sub(r"-  -", "", phrase)
        phrase = re.sub(r"``", "", phrase)
        phrase = re.sub(r":", "", phrase)
        phrase = [func(w) for w in word_tokenize(phrase.lower())]
        phrase = ' '.join(phrase)
        phrases_corpus.append(phrase)
    return phrases_corpus


filename = 'moview_review.tsv'

names = ['PhraseId', 'SentenceId', 'Phrase', 'Sentiment']
dataset = read_csv(filename, names=names, sep='\t')

dataset['clean_review'] = clean_phrases(dataset.Phrase.values)
X = dataset.clean_review
Y = dataset.Sentiment

vect = CountVectorizer()
tfidf = TfidfTransformer()
clf = LinearSVC()

pipeline = Pipeline([('vect', vect), ('tfidf', tfidf), ('clf', clf)])

pipeline.fit(X, Y)

y_pred = pipeline.predict(X)

print(f1_score(Y, y_pred, average='micro'))
