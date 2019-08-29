import pandas

df = pandas.read_csv('train.csv')

print(df.sample(3))
print('Other sample')
print(df.sample(3))
