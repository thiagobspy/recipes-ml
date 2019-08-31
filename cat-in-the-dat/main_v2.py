import pandas
from keras import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

'''
Pego todas as colunas do dataset e aplicado dummies

0.79
'''

pandas.set_option('display.width', 300)
pandas.set_option('display.max_columns', 25)
pandas.set_option('precision', 3)

df_train = pandas.read_csv('train.csv')
df_test = pandas.read_csv('test.csv')

# df_train = pandas.concat([df_train[df_train.target == 0].sample(90000), df_train[df_train.target == 1].sample(90000)])
Y = df_train.target

df = pandas.concat([df_train, df_test])

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

new_df_without_target = df.drop(columns=['target', 'id', 'nom_9'])
df_str = new_df_without_target.astype(str)

X = pandas.get_dummies(df_str)
X_train = X.iloc[:300000]
X_test = X.iloc[300000:]

X_train, X_val, y_train, y_val = train_test_split(X_train, Y, test_size=0.33, random_state=42)

model = Sequential()
model.add(Dense(units=512, input_dim=X.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

checkpoint = ModelCheckpoint(filepath='cat.model.best.hdf5', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, verbose=1, patience=2, min_lr=0.00000001)

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs=50, validation_data=(X_val, y_val),
          callbacks=[checkpoint, reduce_lr])

data_final = pandas.read_csv('sample_submission.csv')
data_final['target'] = model.predict(X_test)
data_final.to_csv('final.csv', index=False)
