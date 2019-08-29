import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

pandas.set_option('display.width', 300)
pandas.set_option('display.max_columns', 25)
pandas.set_option('precision', 3)

df = pandas.read_csv('train.csv')
df_test = pandas.read_csv('train.csv')

df_train = pandas.concat([df[df.target == 0].sample(90000), df[df.target == 1].sample(90000)])

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

new_df_without_target = df.drop(columns=['target'])

df = pandas.get_dummies(new_df_without_target)
df_train = df.iloc[:180000]
df_test = df.iloc[180000:]


# scaler = StandardScaler()
# nom5 = df['nom_5'].apply(lambda x: int(x, 16)).values.reshape(180000, 1)
# scaler.fit(nom5)
# df['nom_5'] = scaler.transform(nom5)
#
# nom6 = df['nom_6'].apply(lambda x: int(x, 16)).values.reshape(180000, 1)
# scaler.fit(nom6)
# df['nom_6'] = scaler.transform(nom6)
#
# nom7 = df['nom_7'].apply(lambda x: int(x, 16)).values.reshape(180000, 1)
# scaler.fit(nom7)
# df['nom_7'] = scaler.transform(nom7)
#
# nom8 = df['nom_8'].apply(lambda x: int(x, 16)).values.reshape(180000, 1)
# scaler.fit(nom8)
# df['nom_8'] = scaler.transform(nom8)

# nom9 = df['nom_9'].apply(lambda x: int(x, 16)).values.reshape(300000, 1)
# scaler.fit(nom9)
# df['nom_9'] = scaler.transform(nom9)

# columns = datadict[datadict['NUnique'] <= 26].index
# new_df = df[columns]
# df_str = new_df_without_target.astype(str)


model = LogisticRegression()
model.fit(df_train, Y)
