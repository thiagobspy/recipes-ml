import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

train = pd.read_csv('train.csv')

Y_train = train['label']
train = train.drop(columns=['label'])

test = pd.read_csv('test.csv')

both = pd.concat([train, test])
both = both / 255

X_train = both.iloc[:train.shape[0], :]
test_final = both.iloc[train.shape[0]:, :]

print(Y_train.value_counts())

Y_train = to_categorical(Y_train, num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.01, random_state=42)

X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)
test_final = test_final.values.reshape(-1, 28, 28, 1)

# define the model
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=192, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=192, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2, padding='same'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, verbose=1, patience=2, min_lr=0.00000001)
model.fit(X_train, y_train, batch_size=128, epochs=25, validation_data=(X_test, y_test),
          callbacks=[checkpointer, reduce_lr])

model.summary()

predict = model.predict(test_final)

data_final = pd.read_csv('sample_submission.csv')
data_final['Label'] = np.argmax(predict, axis=1)
data_final.to_csv('final.csv', index=False)
