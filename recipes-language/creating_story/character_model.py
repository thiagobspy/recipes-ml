import random
import string
import re
import sys

import numpy as np
from os import listdir

from keras import Sequential
from keras.callbacks import LambdaCallback
from keras.layers import LSTM, Dense
from keras.optimizers import RMSprop
from keras.utils import to_categorical, plot_model
from keras_preprocessing.sequence import pad_sequences


def read_files(directory):
    files = []
    for filename in listdir(directory):
        path = directory + '/' + filename
        with open(path, 'r') as file:
            text = file.read()
            files.append(text)
    return files


def clean_text(doc):
    tokens = doc.lower()
    tokens = tokens.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    return ' '.join(tokens)


musics = read_files('livros')
musics = [clean_text(music) for music in musics]
raw_text = ' '.join(musics)

maxlength = 40
step = 5
sequences = []
for i in range(maxlength, len(raw_text), step):
    seq = raw_text[i - maxlength: i + 1]
    sequences.append(seq)

print('Total Sequences: %d' % len(sequences))

chars = sorted(list(set(raw_text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

encoded_sequences = []
for line in sequences:
    encoded_seq = [char_indices[c] for c in line]
    encoded_sequences.append(encoded_seq)

vocab_size = len(char_indices)
print('Vocabulary Size: %d' % vocab_size)

data = np.asarray(encoded_sequences)
X, y = data[:, :-1], data[:, -1]

X_one_hot = np.asarray([to_categorical(x, num_classes=vocab_size) for x in X])
y_one_hot = to_categorical(y, num_classes=vocab_size)

model = Sequential()
model.add(LSTM(128, input_shape=(X_one_hot.shape[1], X_one_hot.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()


# plot_model(model, to_file='model_char.png', show_shapes=True)

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(raw_text) - maxlength - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = raw_text[start_index: start_index + maxlength]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(100):
            x_pred = np.zeros((1, maxlength, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(X_one_hot, y_one_hot, epochs=60, callbacks=[print_callback])


def generate_seq(model, char_indices, seq_length, seed_text, n_chars):
    in_text = seed_text

    for _ in range(n_chars):
        encoded = [char_indices[char] for char in in_text]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        encoded = to_categorical(encoded, num_classes=len(char_indices))
        yhat = model.predict_classes(encoded)
        in_text += indices_char[yhat[0]]

    return in_text


generate_seq(model, char_indices, maxlength, 'so por uma noite', 100)
