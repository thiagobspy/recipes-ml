from keras import Sequential
from keras.preprocessing.text import one_hot
from keras.layers import Embedding, Flatten, Dense
from keras_preprocessing.sequence import pad_sequences

docs = ['Well done!', 'Good work',
        'Great effort', 'nice work', 'Excellent!', 'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']

labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)

max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.summary()

model.fit(padded_docs, labels, epochs=50, verbose=0)

loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy * 100))
