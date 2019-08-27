import pandas

pandas.set_option('display.width', 200)
pandas.set_option('display.max_columns', 15)
pandas.set_option('precision', 3)

df = pandas.read_csv('train.csv').drop(columns='PassengerId')

print(df.head())
