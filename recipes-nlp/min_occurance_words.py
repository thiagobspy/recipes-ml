from collections import Counter

text = ["What is Lorem Ipsum?",
        "Lorem Ipsum is simply dummy text of the printing and typesetting industry.",
        "Lorem Ipsum has been the industry's standard dummy text ever since the 1500",
        "when an unknown printer took a galley of type and scrambled it to make a type specimen book.",
        "It has survived not only five centuries, but also the leap into electronic typesetting"
        ]
vocab = Counter()
for t in text:
    vocab.update(t.split())

print(len(vocab))
min_occurance = 2
tokens = [k for k, c in vocab.items() if c >= min_occurance]
print(len(tokens))
