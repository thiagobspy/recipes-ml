from matplotlib import pyplot
from numpy import mean, std, delete, savetxt
from pandas import read_csv

data = read_csv('eeg_eye_state.csv')
values = data.values
pyplot.figure()

for i in range(values.shape[1] - 1):
    data_mean, data_std = mean(values[:, i]), std(values[:, i])
    cut_off = data_std * 4
    lower, upper = data_mean - cut_off, data_mean + cut_off
    too_small = [j for j in range(values.shape[0]) if values[j, i] < lower]
    values = delete(values, too_small, 0)
    too_large = [j for j in range(values.shape[0]) if values[j, i] > upper]
    values = delete(values, too_large, 0)

savetxt('eeg_eye_state_no_outliers.csv', values, delimiter=',')

for i in range(values.shape[1]):
    pyplot.subplot(values.shape[1], 1, i + 1)
    pyplot.plot(values[:, i])

pyplot.show()
