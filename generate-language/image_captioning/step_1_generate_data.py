import re
import string
from os import listdir, path
from pickle import dump

from keras import Input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input, VGG19
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras_preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from numpy import array


def extract_features(dir):
    in_layer = Input(shape=(224, 224, 3))
    model = VGG19(include_top=False, input_tensor=in_layer, pooling='avg')
    model.summary()

    _images = dict()

    for name in listdir(dir):
        filename = path.join(dir, name)

        image = img_to_array(load_img(filename, target_size=(224, 224)))

        h, w, c = image.shape
        image = image.reshape((1, h, w, c))

        image = preprocess_input(image)
        feature = model.predict(image)

        image_id = name.split('.')[0]

        _images[image_id] = feature

    return _images


def execute_features():
    images = extract_features('dataset/imagens')
    dump(images, open('files/features.pkl', 'wb'))


def clean_text(sentence):
    stop_words = set(stopwords.words('english'))
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    sentence = re_punc.sub('', sentence)
    sentence = re.sub(r"\d", " ", sentence)
    array_string = sentence.split(' ')
    array_string = [word.lower() for word in array_string if word.isalpha()]
    array_string = [word for word in array_string if word not in stop_words]
    array_string = [word for word in array_string if len(word) > 2]

    return ' '.join(array_string)


def load_description(filename):
    mapping = dict()

    file = open(filename, 'r')
    doc = file.read()
    file.close()

    for line in doc.split('\n'):
        tokens = line.split()

        if len(tokens) < 2:
            continue

        image_id = tokens[0].split('.')[0]
        _description = ' '.join(tokens[1:])

        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(clean_text(_description))

    return mapping


def execute_descriptions():
    _descriptions = load_description('dataset/Flickr8k.lemma.token.txt')
    all_tokens = ' '.join([' '.join(desc) for desc in _descriptions.values()]).split(' ')
    print('Words: ', len(all_tokens))
    print('Unique words: ', len(set(all_tokens)))

    return set(all_tokens), _descriptions


def save_descriptions(_descriptions, filename):
    lines = list()
    for key, desc_list in _descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


vocabulary, descriptions = execute_descriptions()
save_descriptions(descriptions, 'files/descriptions.txt')

tokenizer = Tokenizer()
tokenizer.fit_on_texts([' '.join(desc) for desc in descriptions.values()])
vocab_size = len(tokenizer.word_index) + 1

flat_description = [desc for many_desc in descriptions.values() for desc in many_desc]
max_length = max(len(s.split(' ')) for s in flat_description)
print('Max description length: ', max_length)

for key, value in descriptions.items():
    descriptions[key] = pad_sequences(tokenizer.texts_to_sequences(value), maxlen=max_length, padding='post')

dump(descriptions, open('files/descriptions_tokenized.pkl', 'wb'))

padded = [desc for many_desc in descriptions.values() for desc in many_desc]

X = list()
y = list()

for (img_no, seq) in enumerate(padded):
    for i in range(1, len(seq)):
        in_seq = pad_sequences([seq[:i]], maxlen=max_length)[0]
        out_seq = to_categorical([seq[i]], num_classes=vocab_size)[0]
        X.append(in_seq)
        y.append(out_seq)

X, y = array(X), array(y)
print(X.shape)
print(y.shape)
