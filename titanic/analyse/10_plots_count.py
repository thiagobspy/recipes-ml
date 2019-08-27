import matplotlib.pyplot as plt
import seaborn as sns
import pandas

df = pandas.read_csv('train.csv').drop(columns='PassengerId')

fig, axes = plt.subplots(3, 3, figsize=(16, 10))
sns.countplot('Survived', data=df, ax=axes[0, 0])
sns.countplot('Pclass', data=df, ax=axes[0, 1])
sns.countplot('Sex', data=df, ax=axes[0, 2])
sns.countplot('SibSp', data=df, ax=axes[1, 0])
sns.countplot('Parch', data=df, ax=axes[1, 1])
sns.countplot('Embarked', data=df, ax=axes[1, 2])
sns.distplot(df['Fare'], kde=True, ax=axes[2, 1])
sns.distplot(df['Age'].dropna(), kde=True, ax=axes[2, 2])

plt.show()