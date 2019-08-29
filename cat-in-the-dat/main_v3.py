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
# df_test = pandas.read_csv('test.csv')

df = pandas.concat([df_train[df_train.target == 0].sample(1000), df_train[df_train.target == 1].sample(1000)])
Y = df.target

# df = pandas.concat([df_train, df_test])

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
X_train = X
# X_test = X.iloc[300000:]

X_train, X_val, y_train, y_val = train_test_split(X_train, Y, test_size=0.33, random_state=42)

params = dict(
    optimizers=[
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
        'mean_absolute_error',
        'mean_absolute_percentage_error',
        'mean_squared_logarithmic_error',
        'squared_hinge',
        'hinge',
        'categorical_hinge',
        'logcosh',
        'binary_crossentropy',
        'kullback_leibler_divergence',
        'poisson',
        'cosine_proximity'
    ],
    init_mode=[
        Zeros(),
        Ones(),
        RandomNormal(mean=0.0, stddev=0.05, seed=None),
        RandomUniform(minval=-0.05, maxval=0.05, seed=None),
        TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
        VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None),
        Orthogonal(gain=1.0, seed=None),
        Identity(gain=1.0),
        lecun_uniform(seed=None),
        glorot_normal(seed=None),
        glorot_uniform(seed=None),
        he_normal(seed=None),
        lecun_normal(seed=None),
        he_uniform(seed=None),

    ],
    epochs=[15, 30, 50],
    batch=[128, 1024, 4096],
    neuros=[1000, 500, 200],
    dropout=[0.2, 0.35, 0.5],
    layers=[1, 3, 5]
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


model = KerasClassifier(build_fn=create_model)

grid = GridSearchCV(estimator=model, param_grid=params, cv=5, n_jobs=-1)

grid_result = grid.fit(X_train, y_train, validation_data=(X_val, y_val))

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

data_final = pandas.read_csv('sample_submission.csv')
data_final['target'] = grid.predict_proba(X_train)[:, 1]
data_final.to_csv('final.csv', index=False)
