from pickle import load

from keras import Input, Model
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import Dropout, Dense, Embedding, LSTM, add
from keras.utils import to_categorical, plot_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input, VGG19
from keras_preprocessing.image import img_to_array, load_img
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from nltk.translate.bleu_score import corpus_bleu
from numpy import array, argmax


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def load_set(filename):
    doc = load_doc(filename)
    dataset = set()

    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.add(identifier)
    return dataset


def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)

    _descriptions = dict()

    for line in doc.split('\n'):

        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]

        if image_id not in dataset:
            continue

        if image_id not in _descriptions:
            _descriptions[image_id] = list()

        desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
        _descriptions[image_id].append(desc)
    return _descriptions


def load_photo_features(filename, dataset):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}
    return features


def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def create_sequences(tokenizer, max_length, descriptions, photos):
    X1, X2, y = list(), list(), list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]

            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)

    return array(X1), array(X2), array(y)


def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(512,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # plot_model(model, to_file='model.png', show_shapes=True)

    return model


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'

    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)

        if word is None:
            break

        in_text += ' ' + word

        if word == 'endseq':
            break

    return in_text


def cleanup_summary(summary):
    index = summary.find(' startseq ')
    if index > -1:
        summary = summary[len(' startseq '):]
    index = summary.find(' endseq ')
    if index > -1:
        summary = summary[:index]
    return summary


def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    for key, desc_list in descriptions.items():
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        yhat = cleanup_summary(yhat)
        references = [cleanup_summary(d).split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())

    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


filename = 'dataset/Flickr_8k.trainImages.txt'
train = load_set(filename)
print(' Dataset: %d ' % len(train))

filename = 'files/descriptions.txt'
train_descriptions = load_clean_descriptions(filename, train)
print(' Descriptions: train=%d ' % len(train_descriptions))

filename = 'files/features.pkl'
train_features = load_photo_features(filename, train)
print(' Photos: train=%d ' % len(train_features))

tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print(' Vocabulary Size: %d ' % vocab_size)

max_length = max_length(train_descriptions)
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features)

filename = 'dataset/Flickr_8k.testImages.txt'
test = load_set(filename)
print(' Dataset: %d ' % len(test))

filename = 'files/descriptions.txt'
test_descriptions = load_clean_descriptions(filename, test)
print(' Descriptions: test=%d ' % len(test_descriptions))

filename = 'files/features.pkl'
test_features = load_photo_features(filename, test)
print(' Photos: test=%d ' % len(test_features))

X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features)

model = define_model(vocab_size, max_length)

checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, mode='min')

model.fit([X1train, X2train], ytrain, epochs=20, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))

filename = 'model.h5'
model = load_model(filename)

evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)

yhat = generate_desc(model, tokenizer, test_features['3224227640_31865b3651'], max_length)
print(yhat)

in_layer = Input(shape=(224, 224, 3))
vgg19 = VGG19(include_top=False, input_tensor=in_layer, pooling='avg')
vgg19.summary()


def new_description(path_photo, model_img):
    image = img_to_array(load_img(path_photo, target_size=(224, 224)))

    h, w, c = image.shape
    image = image.reshape((1, h, w, c))

    image = preprocess_input(image)
    feature = model_img.predict(image)

    yhat = generate_desc(model, tokenizer, feature, max_length)
    print(yhat)


new_description('dataset/dog_eat.jpeg', vgg19)
new_description('dataset/dog_ball.jpeg', vgg19)
