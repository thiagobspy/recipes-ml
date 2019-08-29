import pandas
from keras import Sequential
from keras.layers import Dense
from sklearn.linear_model import LogisticRegression

pandas.set_option('display.width', 300)
pandas.set_option('display.max_columns', 25)
pandas.set_option('precision', 3)

df = pandas.read_csv('train.csv')
df = pandas.concat([df[df.target == 0].sample(90000), df[df.target == 1].sample(90000)])

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

Y = df.target

columns = datadict[datadict['NUnique'] <= 26].index
new_df = df[columns]
new_df_without_target = new_df.drop(columns='target')
df_str = new_df_without_target.astype(str)

X = pandas.get_dummies(df_str)

model = Sequential()
model.add(Dense(units=128, input_dim=X.shape[1], activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, Y, batch_size=128, epochs=50)
