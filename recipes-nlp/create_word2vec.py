import re
import string
import multiprocessing
import pandas as pd
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from nltk.corpus import stopwords

df = pd.read_csv('./simpsons_dataset.csv')
print(df.shape)
print(df.head())

df.dropna(inplace=True)


def clean_doc(doc):
    doc = doc.lower()
    tokens = doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


text_cleaned = df.apply(lambda x: clean_doc(x['spoken_words']), axis=1)

cores = multiprocessing.cpu_count()
print(cores)
model = Word2Vec(text_cleaned, min_count=10, workers=cores)

# Otimizar memoria porem nao pode mais se treinado
model.wv.init_sims(replace=True)

words = list(model.wv.vocab)
print(words)

print(model['bart'])
model.save('model_simpsons_word2vect.bin')

del model

model = Word2Vec.load('model_simpsons_word2vect.bin')

model.wv.most_similar(positive=["homer"])
model.wv.most_similar(positive=["marge"])
model.wv.most_similar(positive=["bart"])
model.wv.similarity("moe", 'tavern')
model.wv.similarity('maggie', 'baby')
model.wv.similarity('bart', 'nelson')
model.wv.doesnt_match(['jimbo', 'milhouse', 'kearney'])
model.wv.doesnt_match(['bart', 'marge', 'homer', 'lisa', 'magie', 'moe', 'nelson'])
model.wv.doesnt_match(["nelson", "bart", "milhouse"])


