import pandas

df = pandas.read_csv('train.csv').drop(columns='PassengerId')

df['Age'] = pandas.cut(df['Age'], bins=[0, 8, 12, 18, 30, 50, 100], labels=False)
df['Age'] = df['Age'].fillna(-1)
print(df['Age'].value_counts())

