from keras import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import numpy as np

docs = ['Well done!', 'Good work',
        'Great effort', 'nice work', 'Excellent!', 'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']

labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1

encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)

max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# Filtrando apenas as word que estão presentes no nosso dicionario
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

model = Sequential()
model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.summary()

model.fit(padded_docs, labels, epochs=50, verbose=0)

loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy * 100))
