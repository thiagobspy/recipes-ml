import pandas
from keras import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit

'''
Pego todas as colunas do dataset (inclusive norm_9), transformando todas colunas bin em binarias.

loss 0.48507
'''

pandas.set_option('display.width', 300)
pandas.set_option('display.max_columns', 25)
pandas.set_option('precision', 3)

df_train = pandas.read_csv('train.csv')
df_test = pandas.read_csv('test.csv')

# df_train = pandas.concat([df_train[df_train.target == 0].sample(90000), df_train[df_train.target == 1].sample(90000)])
Y = df_train.target

df = pandas.concat([df_train, df_test])
del df_train
del df_test

df['bin_3'] = df['bin_3'].map({'T': 1, 'F': 0})
df['bin_4'] = df['bin_4'].map({'Y': 1, 'N': 0})
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

new_df_without_target = df.drop(columns=['target'])
del df

X = pandas.get_dummies(new_df_without_target, columns=new_df_without_target.columns, drop_first=True, sparse=True)
del new_df_without_target

X_train = X.iloc[:300000].sparse.to_coo().tocsr()
X_test = X.iloc[300000:].sparse.to_coo().tocsr()
del X

X_train, X_val, y_train, y_val = train_test_split(X_train, Y, test_size=0.05, random_state=42)

params = dict(
    optimizer=[
        'sgd',
    ],
    loss=[
        'binary_crossentropy',
    ],
    init_mode=[
        'glorot_uniform'
    ],
    activation=[
        'relu',
        'tanh'
    ],
    epochs=[40],
    batch_size=[32, 128],
    neuros=[
        (128,),
    ],
    dropout=[0.15],
)


def create_model(init_mode, optimizer, loss, neuros, dropout, activation):
    print('Strating...', )
    print('Parans: ', init_mode, optimizer, loss, neuros, dropout, activation)

    model = Sequential()
    first = True
    for neuro in neuros:
        if first:
            model.add(
                Dense(units=neuro, input_dim=X_train.shape[1], kernel_initializer=init_mode, activation=activation))
            first = False
        else:
            model.add(Dense(units=neuro, kernel_initializer=init_mode, activation=activation))
        model.add(Dropout(dropout))

    model.add(Dense(units=1, kernel_initializer=init_mode, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, verbose=2)
grid = GridSearchCV(estimator=model, param_grid=params, cv=ShuffleSplit(test_size=0.05, n_splits=1, random_state=0))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=6)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=2, min_lr=0.00001)

grid_result = grid.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[reduce_lr, es])

# summarize results
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

# write results
result = pandas.DataFrame({'means': means, 'stds': stds, 'params': params})
result.to_csv('results_grid_activation_unit_dropout.csv')

# predict to submit
data_final = pandas.read_csv('sample_submission.csv')
data_final['target'] = grid.predict_proba(X_test)[:, 1]
data_final.to_csv('final.csv', index=False)
