import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv('train.csv').drop(columns='PassengerId')

group_survived_sex = df.groupby(['Survived', 'Sex'])
group_survived_sex.size().unstack(fill_value=0).plot.bar()
print(group_survived_sex.size())

group_survived_has_cabin_sex = df.groupby(['Survived', 'Pclass', 'Sex'])
group_survived_has_cabin_sex.size().unstack(fill_value=0).plot.bar()
print(group_survived_has_cabin_sex.size())

group_survived_has_cabin_sex = df.groupby(['Survived', 'Sex', 'Pclass'])
group_survived_has_cabin_sex.size().unstack(fill_value=0).plot.bar()
print(group_survived_has_cabin_sex.size())

plt.show()
