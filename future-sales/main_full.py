import pandas
from itertools import product

"""
Tentamos colocar item_id e count, porem foi horrivel, nao terminado o teste e descartado.
"""
df_cat = pandas.read_csv('item_categories.csv')
df_item = pandas.read_csv('items.csv')
df_sales = pandas.read_csv('sales_train.csv')
df_shop = pandas.read_csv('shops.csv')
df_sub = pandas.read_csv('sample_submission.csv')
df_test = pandas.read_csv('test.csv')

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
