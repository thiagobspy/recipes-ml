learning = 0.1
decay_rate = 0.03


def apply_decay(learning, decay_rate, epoch):
    return 1 / (1 + decay_rate * epoch) * learning


for i in range(5):
    learning = apply_decay(learning, decay_rate, i)
    print(learning)
