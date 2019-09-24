from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot

model = Word2Vec.load('model_simpsons_word2vect.bin')
words = ['taking', 'mrs', 'bart', 'homer', 'car', 'food', 'ball', 'lisa', 'sometimes', 'burns', 'sure', 'like']

X = model[words]

pca = PCA(n_components=2)
result = pca.fit_transform(X)
pyplot.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
