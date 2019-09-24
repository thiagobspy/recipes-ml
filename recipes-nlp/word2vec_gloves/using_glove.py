from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# http://nlp.stanford.edu/data/glove.6B.zip

glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=True)
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)
