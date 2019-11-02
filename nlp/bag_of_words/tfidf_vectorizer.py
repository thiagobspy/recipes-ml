"""
Semelhante ao counter, porem no lugar de coloca a quantidade repetições, é calculado um valor que representa um score para cada palavra,
assim diminui a importancia de palavras muito repetidas e sem relevancia.
"""
from sklearn.feature_extraction.text import TfidfVectorizer

text = ["What is Lorem Ipsum?",
        "Lorem Ipsum is simply dummy text of the printing and typesetting industry.",
        "Lorem Ipsum has been the industry's standard dummy text ever since the 1500",
        "when an unknown printer took a galley of type and scrambled it to make a type specimen book.",
        "It has survived not only five centuries, but also the leap into electronic typesetting"
        ]
vectorizer = TfidfVectorizer()
vectorizer.fit(text)

print(vectorizer.vocabulary_)
print(vectorizer.idf_)

vector = vectorizer.transform(text)

print(vector.shape)
print(type(vector))
print(vector.toarray())
