import pandas
from itertools import product

df_cat = pandas.read_csv('item_categories.csv')
df_item = pandas.read_csv('items.csv')
df_sales = pandas.read_csv('sales_train.csv')
df_shop = pandas.read_csv('shops.csv')
df_sub = pandas.read_csv('sample_submission.csv')
df_test = pandas.read_csv('test.csv')

sales_group = df_sales.groupby(['shop_id', 'date_block_num', 'item_id'])

df_sales = pandas.DataFrame({'count': sales_group.size()}).reset_index()

shops_train = set(df_sales.shop_id)
shops_test = set(df_sales.shop_id)

item_train = set(df_sales.item_id)
item_test = set(df_sales.item_id)

df = pandas.DataFrame(product(df_shop.shop_id, df_item.item_id), columns=['shop_id', 'item_id'])
