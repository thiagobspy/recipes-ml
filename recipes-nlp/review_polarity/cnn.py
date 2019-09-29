import re
import string
from collections import Counter
from os import listdir

import nltk
import pandas
from keras import Sequential, Input, Model
from keras.layers import Dense, Embedding, Conv1D, Dropout, MaxPooling1D, Flatten, concatenate
from keras.utils import plot_model
from keras_preprocessing.sequence import pad_sequences
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
vocab_size = len(tokens) + 1

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
encoded = tokenizer.texts_to_sequences(df['text'])
length = max([len(s) for s in encoded])
padded = pad_sequences(encoded, maxlen=length, padding='post')

# channel 1
inputs1 = Input(shape=(length,))
embedding1 = Embedding(vocab_size, 100)(inputs1)
conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
drop1 = Dropout(0.5)(conv1)
pool1 = MaxPooling1D(pool_size=2)(drop1)
flat1 = Flatten()(pool1)

# channel 2
inputs2 = Input(shape=(length,))
embedding2 = Embedding(vocab_size, 100)(inputs2)
conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
drop2 = Dropout(0.5)(conv2)
pool2 = MaxPooling1D(pool_size=2)(drop2)
flat2 = Flatten()(pool2)

# channel 3
inputs3 = Input(shape=(length,))
embedding3 = Embedding(vocab_size, 100)(inputs3)
conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
drop3 = Dropout(0.5)(conv3)
pool3 = MaxPooling1D(pool_size=2)(drop3)
flat3 = Flatten()(pool3)

# merge
merged = concatenate([flat1, flat2, flat3])

# interpretation
dense1 = Dense(10, activation='relu')(merged)
outputs = Dense(1, activation='sigmoid')(dense1)
model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# summarize

model.summary()

model.fit([padded, padded, padded], df['target'], epochs=7, batch_size=16)


def whats_sentiment(text):
    text = ' '.join(clean_doc(text))
    x = tokenizer.texts_to_matrix([text], mode='binary')
    return model.predict(x)
