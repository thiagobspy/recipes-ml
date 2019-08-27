import pandas as pd
import numpy
from keras import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('train.csv')

Y_train = train_df['label']
Y_train = to_categorical(Y_train, num_classes=10)
X_train = train_df.drop(columns=['label'])

X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)

X_train = X_train / 255

model = Sequential()
model.add(Dense(units=32, input_shape=[784, ], activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_test, y_test))

result = model.predict(X_test)

from_result = numpy.argmax(result, axis=1)
from_test = numpy.argmax(y_test, axis=1)

sum(from_test == from_result) / len(from_result)
