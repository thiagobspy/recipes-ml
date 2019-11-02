import re
import string
from collections import Counter
from os import listdir

import nltk
import pandas
from keras import Sequential
from keras.layers import Dense
from keras_preprocessing.text import Tokenizer
from matplotlib import pyplot
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download('stopwords')


def process_docs(directory, vocab):
    lines = []
    for filename in listdir(directory):
        if not filename.endswith('.txt'):
            continue
        path = directory + '/' + filename
        doc = load_doc(path)
        doc_cleaned = clean_doc(doc)
        vocab.update(doc_cleaned)
        lines.append(' '.join(doc_cleaned))
    return lines


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()

    return text


def clean_doc(doc):
    stemmer = SnowballStemmer('english')

    doc = doc.split()

    re_punc = re.compile('[%s]' % re.escape(string.punctuation))

    doc = [re_punc.sub('', word) for word in doc]

    doc = [word for word in doc if word.isalpha()]

    stop_words = set(stopwords.words('english'))

    doc = [word for word in doc if word not in stop_words]

    doc = [word for word in doc if len(word) > 1]

    doc = [stemmer.stem(word) for word in doc]

    return doc


def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def doc_to_line(filename):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    return ' '.join(tokens)


################################# START ################################

vocab = Counter()
negative_lines = process_docs('txt_sentoken/neg', vocab)
positive_lines = process_docs('txt_sentoken/pos', vocab)

print(len(vocab))
print(vocab.most_common(50))

min_occurance = 2
tokens = set([k for k, c in vocab.items() if c >= min_occurance])
print(len(tokens))

df_negative = pandas.DataFrame(negative_lines, columns=['text'])
df_negative['target'] = 0

df_positive = pandas.DataFrame(positive_lines, columns=['text'])
df_positive['target'] = 1

df = pandas.concat([df_negative, df_positive], axis=0)

df['text'] = df.apply(lambda x: ' '.join([word for word in x['text'].split() if word in tokens]), axis=1)

modes = ['binary', 'count', 'tfidf', 'freq']
results = pandas.DataFrame()

for mode in modes:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['text'])
    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.10, random_state=42)
    x_train = tokenizer.texts_to_matrix(x_train, mode=mode)
    x_test = tokenizer.texts_to_matrix(x_test, mode=mode)

    scores = list()
    n_repeats = 10
    n_words = x_train.shape[1]
    for i in range(n_repeats):
        model = Sequential()
        model.add(Dense(50, input_shape=(n_words,), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=10, verbose=0)

        _, acc = model.evaluate(x_test, y_test, verbose=0)
        scores.append(acc)
    results[mode] = scores

results.boxplot()
pyplot.show()
print(results)


def whats_sentiment(text):
    text = ' '.join(clean_doc(text))
    x = tokenizer.texts_to_matrix([text], mode='binary')
    return model.predict(x)
