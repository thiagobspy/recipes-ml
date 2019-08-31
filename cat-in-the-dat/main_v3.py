import pandas
from keras import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.initializers import Zeros, lecun_uniform, Identity, Orthogonal, VarianceScaling, TruncatedNormal, \
    RandomUniform, RandomNormal, Ones, he_uniform, lecun_normal, he_normal, glorot_uniform, glorot_normal
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Nadam, Adamax
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

X_train, X_val, y_train, y_val = train_test_split(X_train, Y, test_size=0.33, random_state=42)

params = dict(
    optimizer=[
        SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False),
        RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),
        Adagrad(lr=0.01, epsilon=None, decay=0.0),
        Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
        Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
        Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0),
        Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    ],
    loss=[
        'mean_squared_error',
        'binary_crossentropy',
    ],
    init_mode=[
        Zeros(),
        RandomNormal(mean=0.0, stddev=0.05, seed=None),
        RandomUniform(minval=-0.05, maxval=0.05, seed=None),
    ],
    epochs=[15, 50],
    batch_size=[1024, 4096],
    neuros=[700, 200],
    dropout=[0.3, 0.5],
    layers=[2, 5]
)


def create_model(init_mode, optimizer, loss, neuros, dropout, layers):
    model = Sequential()
    model.add(Dense(units=neuros, input_dim=X.shape[1], kernel_initializer=init_mode, activation='relu'))
    model.add(Dropout(dropout))

    for i in range(layers):
        model.add(Dense(units=int(neuros / (i + 1)), kernel_initializer=init_mode, activation='relu'))
        model.add(Dropout(dropout))

    model.add(Dense(units=1, kernel_initializer=init_mode, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


model = KerasClassifier(build_fn=create_model, verbose=2)

grid = GridSearchCV(estimator=model, param_grid=params, cv=3, n_jobs=1)

grid_result = grid.fit(X_train, y_train, validation_data=(X_val, y_val))

# summarize results
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

# write results
result = pandas.DataFrame({'means': means, 'stds': stds, 'params': params})
result.to_csv('results_grid.csv')

# predict to submit
data_final = pandas.read_csv('sample_submission.csv')
data_final['target'] = grid.predict_proba(X_train)[:, 1]
data_final.to_csv('final.csv', index=False)
