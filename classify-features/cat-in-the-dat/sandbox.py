import pandas
from keras import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit

'''

'''

pandas.set_option('display.width', 300)
pandas.set_option('display.max_columns', 25)
pandas.set_option('precision', 3)

df_train = pandas.read_csv('train.csv')
df_test = pandas.read_csv('test.csv')

Y = df_train.target

df = pandas.concat([df_train, df_test])
del df_train
del df_test

df['bin_3'] = df['bin_3'].map({'T': 1, 'F': 0})
df['bin_4'] = df['bin_4'].map({'Y': 1, 'N': 0})
df.drop(columns=['id'], inplace=True)

columns = df.columns

datas = [
    ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4'],
    ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'day', 'month'],
    ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5'],
    ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month'],
    ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'day', 'month', 'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5'],
    ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
]
register = []
for data in datas:
    new_df_without_target = df[data]
    X = pandas.get_dummies(new_df_without_target, columns=new_df_without_target.columns, drop_first=True, sparse=True)
    X_train = X.iloc[:300000].sparse.to_coo().tocsr()
    X_test = X.iloc[300000:].sparse.to_coo().tocsr()

    X_train, X_val, y_train, y_val = train_test_split(X_train, Y, test_size=0.05, random_state=42)

    model = Sequential()
    model.add(Dense(units=128, input_dim=X_train.shape[1], activation='tanh'))
    model.add(Dropout(0.35))
    model.add(Dense(units=1, activation='sigmoid'))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
    checkpoint = ModelCheckpoint(filepath='cat.model.best.hdf5', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, verbose=1, patience=3, min_lr=0.0001)

    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    fitted = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_val, y_val),
                       callbacks=[checkpoint, reduce_lr, es])
    register.append(fitted.history)
