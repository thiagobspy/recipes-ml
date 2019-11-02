import re
import string
from unidecode import unidecode

from nltk import word_tokenize, SnowballStemmer
from nltk.corpus import stopwords


def remove_punctuation(sentence):
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    return re_punc.sub('', sentence)


def remove_accentuation(sentence):
    return unidecode(sentence)


def remove_digits(sentence):
    return re.sub(r"\d", " ", sentence)


def choose_only_string(array_string):
    return [word for word in array_string if word.isalpha()]


def using_stopwords(array_string):
    stop_words = set(stopwords.words('portuguese'))
    return [word for word in array_string if word not in stop_words]


def using_stemmer(array_string):
    stemmer = SnowballStemmer('portuguese')
    return [stemmer.stem(word) for word in array_string]


def remove_len_less(array_string):
    return [word for word in array_string if len(word) > 2]


sentence = 'Points! It`s car!_ it cão is ok estamos 31'
print(remove_punctuation(sentence))
print(remove_accentuation(sentence))
print(remove_digits(sentence))
print(choose_only_string(sentence.split(' ')))
print(using_stopwords(sentence.split(' ')))
print(using_stemmer(sentence.split(' ')))
print(remove_len_less(sentence.split(' ')))
