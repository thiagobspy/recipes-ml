"""
Semelhante ao Hash comum, é traduzido cada string para uma casa do Hash, passamos na inicialização a quantidade de feature que tera.
"""
from sklearn.feature_extraction.text import HashingVectorizer

text = ["What is Lorem Ipsum?",
        "Lorem Ipsum is simply dummy text of the printing and typesetting industry.",
        "Lorem Ipsum has been the industry's standard dummy text ever since the 1500",
        "when an unknown printer took a galley of type and scrambled it to make a type specimen book.",
        "It has survived not only five centuries, but also the leap into electronic typesetting"
        ]
vectorizer = HashingVectorizer(n_features=20)

vector = vectorizer.transform(text)

print(vector.shape)
print(vector.toarray())
