import pandas
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

'''
Pego todas as colunas do dataset e aplicado dummies
Utilizando GridSearchCV.
'''

pandas.set_option('display.width', 300)
pandas.set_option('display.max_columns', 25)
pandas.set_option('precision', 3)

df_train = pandas.read_csv('train.csv')
df_test = pandas.read_csv('test.csv')

# df = pandas.concat([df_train[df_train.target == 0].sample(1000), df_train[df_train.target == 1].sample(1000)])
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

X_train, X_val, y_train, y_val = train_test_split(X_train, Y, test_size=0.10, random_state=42)

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
    epochs=[40],
    batch_size=[128],
    neuros=[
        (256, 64),
    ],
    dropout=[0.3],
)


def create_model(init_mode, optimizer, loss, neuros, dropout):
    model = Sequential()
    first = True
    for neuro in neuros:
        if first:
            model.add(Dense(units=neuro, input_dim=X.shape[1], kernel_initializer=init_mode, activation='relu'))
            first = False
        else:
            model.add(Dense(units=neuro, kernel_initializer=init_mode, activation='relu'))
        model.add(Dropout(dropout))

    model.add(Dense(units=1, kernel_initializer=init_mode, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, verbose=2)
grid = GridSearchCV(estimator=model, param_grid=params, cv=2, n_jobs=1)

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, verbose=1, patience=2, min_lr=0.00000001)
grid_result = grid.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[reduce_lr])

# summarize results
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

# write results
result = pandas.DataFrame({'means': means, 'stds': stds, 'params': params})
result.to_csv('results_grid_layers.csv')

# predict to submit
data_final = pandas.read_csv('sample_submission.csv')
data_final['target'] = grid.predict_proba(X_test)[:, 1]
data_final.to_csv('final.csv', index=False)
