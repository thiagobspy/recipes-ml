from gensim.models import KeyedVectors

# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)
