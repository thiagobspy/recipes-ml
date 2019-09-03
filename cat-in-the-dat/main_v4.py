import pandas
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, log_loss
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

sample_train = 300000
sample_test = 200000
# df_train = pandas.concat([df_train[df_train.target == 0].sample(sample_train // 2),
#                           df_train[df_train.target == 1].sample(sample_train // 2)])
# df_test = df_test.sample(sample_test)

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


# Meta-transformer for selecting features based on importance weights.
def select_from_model(x, y, test):
    lg = LogisticRegression(solver='lbfgs')
    sfm = SelectFromModel(lg)
    sfm.fit(x, y)
    return 'select_from_model', sfm.transform(x), sfm.transform(test)


def select_k_best(x, y, test):
    skb = SelectKBest(score_func=chi2, k=x.shape[1] // 3)
    skb.fit(x, y)
    return 'select_k_best', skb.transform(x), skb.transform(test)


feature_selections = [
    select_from_model,
    select_k_best,
]

for feature_selection in feature_selections:
    X_train = X.iloc[:sample_train]
    X_test = X.iloc[sample_train:]

    name, X_train, X_test = feature_selection(X_train, Y, X_test)

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
        epochs=[40, 60],
        batch_size=[128],
        neuros=[
            (64, 64),
        ],
        dropout=[0.3],
    )


    def create_model(init_mode, optimizer, loss, neuros, dropout):
        print('Strating...', )
        print('Feature: ', name)
        print('Shape: ', X_train.shape)
        print('Neuros: ', neuros)

        sequential = Sequential()
        first = True
        for neuro in neuros:
            if first:
                sequential.add(
                    Dense(units=neuro, input_dim=X_train.shape[1], kernel_initializer=init_mode, activation='relu'))
                first = False
            else:
                sequential.add(Dense(units=neuro, kernel_initializer=init_mode, activation='relu'))
            sequential.add(Dropout(dropout))

        sequential.add(Dense(units=1, kernel_initializer=init_mode, activation='sigmoid'))
        sequential.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        return sequential


    model = KerasClassifier(build_fn=create_model, verbose=2)
    LogLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
    grid = GridSearchCV(estimator=model, param_grid=params, cv=2, n_jobs=1)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3, verbose=1, patience=2, min_lr=0.00000001)
    grid_result = grid.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[reduce_lr])

    # summarize results
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    # write results
    result = pandas.DataFrame({'means': means, 'stds': stds, 'params': params})
    result.to_csv('results_feature_' + name + '.csv')

    # predict to submit
    data_final = pandas.read_csv('sample_submission.csv')
    data_final['target'] = grid.predict_proba(X_test)[:, 1]
    data_final.to_csv('final' + name + '.csv', index=False)
