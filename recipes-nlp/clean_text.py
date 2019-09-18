import re
import string
from unidecode import unidecode

from nltk import word_tokenize, SnowballStemmer
from nltk.corpus import stopwords


def clean_doc(doc):
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def clean_text(doc):
    stop_words = set(stopwords.words('portuguese') + list(string.punctuation))

    stemmer = SnowballStemmer('portuguese')
    tokens = doc.lower()
    tokens = unidecode(tokens)
    tokens = re.sub(r"/", " / ", tokens)
    tokens = re.sub(r"\d", " ", tokens)
    tokens = [str(w) for w in word_tokenize(tokens)]
    tokens = filter(lambda word: word not in stop_words and len(word) >= 2 and word.isalpha(), tokens)
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)
