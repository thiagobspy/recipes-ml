import pandas
from keras import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

'''
Pego todas as colunas do dataset, transformando todas colunas bin em binarias.
Criado uma rede para cada mes, porem nao foi legal.

'''

pandas.set_option('display.width', 300)
pandas.set_option('display.max_columns', 25)
pandas.set_option('precision', 3)

df_train_all = pandas.read_csv('train.csv')
df_test_all = pandas.read_csv('test.csv')

gp_train = df_train_all.groupby('month')
gp_test = df_test_all.groupby('month')

fits = []

for i in range(1, 13):
    print('=' * 50)
    print('Month: ', 1)
    df_train = gp_train.get_group(i)
    df_test = gp_test.get_group(i)
    Y = df_train.target

    df = pandas.concat([df_train, df_test])

    df['bin_3'] = df['bin_3'].map({'T': 1, 'F': 0})
    df['bin_4'] = df['bin_4'].map({'Y': 1, 'N': 0})
    df.drop(columns=['id', 'month', 'nom_9'], inplace=True)

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

    X = pandas.get_dummies(new_df_without_target, columns=new_df_without_target.columns, drop_first=True, sparse=True)

    X_train = X.iloc[:df_train.shape[0]].sparse.to_coo().tocsr()
    X_test = X.iloc[df_train.shape[0]:].sparse.to_coo().tocsr()

    X_train, X_val, y_train, y_val = train_test_split(X_train, Y, test_size=0.05, random_state=42)

    model = Sequential()
    model.add(Dense(units=128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer='l2'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    checkpoint = ModelCheckpoint(filepath='cat.model.best.hdf5', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, verbose=1, patience=2, min_lr=0.00000001)

    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    fitter = model.fit(X_train, y_train, batch_size=128, epochs=100, validation_data=(X_val, y_val),
                       callbacks=[checkpoint, reduce_lr])

    fits.append((i, fitter))
