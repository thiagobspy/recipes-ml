import random
import string
import re
import sys

import numpy as np
from os import listdir

from keras import Sequential
from keras.callbacks import LambdaCallback
from keras.layers import LSTM, Dense, Embedding
from keras.optimizers import RMSprop
from keras.utils import to_categorical, plot_model
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer


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
    return ' '.join(tokens)


musics = read_files('charlie_brown')
musics = [clean_text(music) for music in musics]
raw_text = ' '.join(musics).split()

word_in = 5
sequences = []
for i in range(word_in, len(raw_text)):
    seq = raw_text[i - word_in:i + 1]
    sequences.append(seq)

print('Total Sequences: %d' % len(sequences))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sequences)
encoded = tokenizer.texts_to_sequences(sequences)

vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

word_indices = tokenizer.word_index
indices_word = dict()
for key, index in tokenizer.word_index.items():
    indices_word[index] = key
indices_word[0] = '_'

data = np.asarray(encoded)
X, y = data[:, :-1], data[:, -1]

y_one_hot = to_categorical(y, num_classes=vocab_size)

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=X.shape[1]))
model.add(LSTM(128))
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

    start_index = random.randint(0, len(raw_text) - word_in - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        sentence = raw_text[start_index: start_index + word_in]
        print('----- Generating with seed: "' + str(sentence) + '"')

        sys.stdout.write(' '.join(sentence) + ' ')

        for i in range(100):
            x_pred = tokenizer.texts_to_sequences([' '.join(sentence)])
            x_pred = np.asarray(x_pred)
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]
            sentence.append(next_word)
            sentence = sentence[1:]

            sys.stdout.write(next_word + ' ')
            sys.stdout.flush()
        print()


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(X, y_one_hot, epochs=60, callbacks=[print_callback])


def generate_seq(model, indices_word, seq_length, seed_text, n_word):
    in_text = [seed_text]

    for _ in range(n_word):
        encoded = tokenizer.texts_to_sequences(in_text)
        encoded = pad_sequences(encoded[-seq_length:], maxlen=seq_length, truncating='pre')
        yhat = model.predict_classes(encoded)
        in_text.append(indices_word[yhat[0]])

    return ' '.join(in_text)


generate_seq(model, indices_word, word_in, 'vida', 100)
