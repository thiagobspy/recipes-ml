from sklearn.feature_extraction.text import CountVectorizer

"""
Cria um one-hot com todas palavras, e o valor é a quantidade de vezes que aquela palavra apareceu no exemplo.
Por conta de existir muito zero, normalmente se trabalha com matrix sparse
"""

text = ["What is Lorem Ipsum?",
        "Lorem Ipsum is simply dummy text of the printing and typesetting industry.",
        "Lorem Ipsum has been the industry's standard dummy text ever since the 1500",
        "when an unknown printer took a galley of type and scrambled it to make a type specimen book.",
        "It has survived not only five centuries, but also the leap into electronic typesetting"
        ]
vectorizer = CountVectorizer()
vectorizer.fit(text)

print(vectorizer.vocabulary_)

vector = vectorizer.transform(text)

print(vector.shape)
print(type(vector))
print(vector.toarray())
