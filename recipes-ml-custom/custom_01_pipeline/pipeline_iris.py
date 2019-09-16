from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

filename = 'iris.data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'target']
dataset = read_csv(filename, names=names)

X = dataset.drop('target', axis=1)
Y = dataset.target

pipeline = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])

pipeline.fit(X, Y)

Y_target = pipeline.predict(X)

print(f1_score(Y, Y_target, average='micro'))
