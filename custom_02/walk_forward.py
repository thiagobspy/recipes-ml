from numpy import array
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = read_csv('eeg_eye_state_no_outliers.csv')
values = data.values
X, y = values[:, :-1], values[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=1)

X_history, y_history = [x for x in X_train], [x for x in y_train]
predictions = list()
for i in range(len(y_test)):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(array(X_history), array(y_history))
    y_pred = model.predict([X_test[i, :]])[0]
    predictions.append(y_pred)
    X_history.append(X_test[i, :])
    y_history.append(y_test[i])
score = accuracy_score(y_test, predictions)
print(score)
