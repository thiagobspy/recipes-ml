import numpy
import pandas
from itertools import product

from keras import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

"""
Tentamos colocar item_id e count, porem foi horrivel, nao terminado o teste e descartado.
"""

df_cat = pandas.read_csv('item_categories.csv')
df_item = pandas.read_csv('items.csv')
df_sales = pandas.read_csv('sales_train.csv')
df_shop = pandas.read_csv('shops.csv')
df_sub = pandas.read_csv('sample_submission.csv')
df_test = pandas.read_csv('test.csv')

df_sales = df_sales[df_sales.shop_id == 2]
sales_group = df_sales.groupby(['shop_id', 'date_block_num', 'item_id'])
sales_sum = sales_group["item_cnt_day"].apply(lambda x: x.sum())
df_sales_sum = pandas.DataFrame(sales_sum.reset_index())

df_sales_count = pandas.DataFrame({'count': sales_group.size()}).reset_index()

shops_train = set(df_sales.shop_id)
shops_test = set(df_test.shop_id)
missing_shop_train = shops_train - shops_test
missing_shop_test = shops_test - shops_train
print('Missing train:', missing_shop_train)
print('Missing test:', missing_shop_test)

item_train = set(df_sales.item_id)
item_test = set(df_test.item_id)
missing_item_train = item_train - item_test
missing_item_test = item_test - item_train
print('Missing train:', missing_item_train)
print('Missing test:', missing_item_test)

shops = shops_train.intersection(shops_test)
items = item_train.intersection(item_test)
count_month = range(0, 34)

df_shop_item = pandas.DataFrame(product(shops, items), columns=['shop_id', 'item_id'])
df_month_item = pandas.DataFrame(product(count_month, items), columns=['date_block_num', 'item_id'])
df_default = pandas.merge(left=df_month_item, right=df_shop_item)

df_full = pandas.merge(left=df_default, right=df_sales_sum, how='left', on=['shop_id', 'item_id', 'date_block_num'])
df_full.fillna(0, inplace=True)
df_full.loc[df_full.item_cnt_day < 0, 'item_cnt_day'] = 0
df_full.item_cnt_day = df_full.item_cnt_day.astype(int)
df_full.rename(columns={'date_block_num': 'month', 'item_cnt_day': 'count'}, inplace=True)

df_full.drop(columns=['shop_id'], inplace=True)
df_full.item_id = df_full.item_id.astype(str)
df = pandas.get_dummies(df_full)

df = df[df.month > 28]
df.drop(columns=['month'], inplace=True)


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence.iloc[i:end_ix], sequence.iloc[end_ix:end_ix + 1]
        X.append(seq_x)
        y.append(seq_y)
    return X, y


X_train, Y_train = split_sequence(df, 3)
X_train = [x.values for x in X_train]
Y_train = [y['count'].values[0] for y in Y_train]

X_train = numpy.array(X_train)
Y_train = numpy.array(Y_train)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])

X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.10, random_state=42)

model = Sequential()
model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=40, batch_size=128, validation_data=(X_val, y_val))
