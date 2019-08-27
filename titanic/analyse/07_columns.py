import pandas

df = pandas.read_csv('train.csv').drop(columns='PassengerId')

print(df.columns)