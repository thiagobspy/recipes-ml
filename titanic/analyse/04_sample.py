import pandas

df = pandas.read_csv('train.csv').drop(columns='PassengerId')

print(df.sample(3))
print('Other sample')
print(df.sample(3))
