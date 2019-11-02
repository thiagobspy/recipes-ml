import pandas
from keras import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit

'''
As colunas nom, tentei da um split, criando outras colunas com apenas um character, porem foi horrivel
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
for i in range(5, 10):
    for j in range(9):
        df['new_nom_' + str(i) + '_' + str(j)] = df['nom_' + str(i)].apply(lambda x: x[j])
df.drop(columns=['id'], inplace=True)

print(df.head())

print(df.shape)

print(df.info())

print(df.sample(3))

datadict = pandas.DataFrame(df.dtypes)
datadict = datadict.rename(columns={0: 'DataType'})
datadict['Count'] = df.count()
datadict['MissingVal'] = df.isnull().sum()
datadict['NUnique'] = df.nunique()
print(datadict)

for column in df.columns:
    print(column)
    print(df[column].value_counts(), end='\n\n\n')

new_df_without_target = df.drop(columns=['target', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'])
del df

X = pandas.get_dummies(new_df_without_target, columns=new_df_without_target.columns, drop_first=True, sparse=True)
del new_df_without_target

X_train = X.iloc[:300000].sparse.to_coo().tocsr()
X_test = X.iloc[300000:].sparse.to_coo().tocsr()
del X

X_train, X_val, y_train, y_val = train_test_split(X_train, Y, test_size=0.05, random_state=42)

model = Sequential()
model.add(Dense(units=256, input_dim=X_train.shape[1], activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(units=512, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(units=128, activation='tanh', kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid', kernel_regularizer='l2'))

checkpoint = ModelCheckpoint(filepath='cat.model.best.hdf5', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, verbose=1, patience=3, min_lr=0.00001)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_val, y_val),
          callbacks=[checkpoint, reduce_lr], verbose=2)

data_final = pandas.read_csv('sample_submission.csv')
data_final['target'] = model.predict(X_test)
data_final.to_csv('final.csv', index=False)
