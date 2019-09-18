import re
import string
from collections import Counter
from os import listdir

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


def process_docs(directory, vocab):
    lines = []
    for filename in listdir(directory):
        if not filename.endswith('.txt'):
            continue
        path = directory + '/' + filename
        add_doc_to_vocab(path, vocab)
        line = doc_to_line(path, vocab)
        lines.append(line)
    return lines


def add_doc_to_vocab(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()

    return text


def clean_doc(doc):
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def doc_to_line(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)


vocab = Counter()
negative_lines = process_docs('txt_sentoken/neg', vocab)
save_list(negative_lines, 'negative.txt')
positive_lines = process_docs('txt_sentoken/pos', vocab)
save_list(positive_lines, 'positive.txt')

print(len(vocab))
print(vocab.most_common(50))

min_occurance = 5
tokens = [k for k, c in vocab.items() if c >= min_occurance]
print(len(tokens))

save_list(tokens, 'vocab.txt')

vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
