from keras_preprocessing.text import text_to_word_sequence, hashing_trick, Tokenizer
from keras.preprocessing.text import one_hot

text = ["What is Lorem Ipsum?",
        "Lorem Ipsum is simply dummy text of the printing and typesetting industry.",
        "Lorem Ipsum has been the industry's standard dummy text ever since the 1500",
        "when an unknown printer took a galley of type and scrambled it to make a type specimen book.",
        "It has survived not only five centuries, but also the leap into electronic typesetting"
        ]

sentences = map(lambda x: text_to_word_sequence(x), text)

words_unique = set()
for words in sentences:
    print(len(words))
    print(words)
    words_unique.update(words)

print('Words uniques: ', len(words_unique))

"""
Semelhante ao Hashing Vectorization, apesar de ter o nome one_hot, na trabalho com o esquema de hash.
Cada casa do vector, fica com um valor de inteiro representando a casa do hash que foi calculado.
"""
hots = map(lambda x: one_hot(x, round(len(words_unique) * 1.3)), text)
for hot in hots:
    print(hot)

hots = map(lambda x: hashing_trick(x, round(len(words_unique) * 1.3)), text)
for hot in hots:
    print(hot)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

print('Palavra e quantidade de cada uma:', tokenizer.word_counts)
print('Quantidade de sentencia:', tokenizer.document_count)
print('Palavra e seu index:', tokenizer.word_index)
print('Palavra e qunatas sentencia apareceu:', tokenizer.word_docs)

tokens = tokenizer.texts_to_sequences(text)
print(tokens)

# Transforma tokens em sentencia novamente
print(tokenizer.sequences_to_texts(tokens))

# Semelhante ao counter_vectorizer
print(tokenizer.texts_to_matrix(text, mode='count'))

# Semelhante ao counter_vectorizer, porem ao em vez de colocar a quantidade de vezes que apareceu, mostra apenas 0 ou 1, se apareceu ou não
print(tokenizer.texts_to_matrix(text, mode='binary'))

# Exatamente igual o processo, unica diferença é quando chama o método text_to_sequence, que ele só considera as 5 primeiras palavras
tokenizer = Tokenizer(num_words=5)
tokenizer.fit_on_texts(text)

print('Palavra e quantidade de cada uma:', tokenizer.word_counts)
print('Quantidade de sentencia:', tokenizer.document_count)
print('Palavra e seu index:', tokenizer.word_index)
print('Palavra e qunatas sentencia apareceu:', tokenizer.word_docs)

tokens = tokenizer.texts_to_sequences(text)
print(tokens)
