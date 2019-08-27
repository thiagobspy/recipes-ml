import pandas

df = pandas.read_csv('train.csv').drop(columns='PassengerId')

correlations = df.drop(columns=['Name', 'Ticket', 'Cabin']).corr(method='pearson')
print(correlations)
