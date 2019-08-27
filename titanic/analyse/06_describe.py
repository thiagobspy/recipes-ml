import pandas

df = pandas.read_csv('train.csv').drop(columns='PassengerId')

print(df.describe(include='all'))

print(df.describe(include='object'))

print(df.describe(include='number'))
