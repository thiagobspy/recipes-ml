from matplotlib import pyplot
from pandas import read_csv

data = read_csv('eeg_eye_state.csv')
values = data.values
pyplot.figure()

for i in range(values.shape[1]):
    pyplot.subplot(values.shape[1], 1, i + 1)
    pyplot.plot(values[:, i])

pyplot.show()
