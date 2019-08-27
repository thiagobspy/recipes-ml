import pandas

df = pandas.read_csv('train.csv').drop(columns='PassengerId')

print(df['Sex'].value_counts())

print(df['Age'].value_counts())

print(df['Survived'].value_counts())

group_survived = df.groupby(['Survived'])
print(group_survived.size())

group_survived_sex = df.groupby(['Survived', 'Sex'])
print(group_survived_sex.size())
