import pandas

df = pandas.read_csv('train.csv')

print(df.describe(include='all'))

print(df.describe(include='object'))

print(df.describe(include='number'))
