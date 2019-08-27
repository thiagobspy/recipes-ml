import pandas

df = pandas.read_csv('train.csv').drop(columns='PassengerId')

df['Sex_cat'] = df['Sex'].astype('category')

print('Default', df['Sex_cat'], sep='\n', end='\n\n\n')
print('Name', df['Sex_cat'].name, sep='\n', end='\n\n\n')
print('Categories', df['Sex_cat'].cat.categories, sep='\n', end='\n\n\n')
print('Codes', df['Sex_cat'].cat.codes, sep='\n', end='\n\n\n')

df['Sex_cat'].cat.categories = ['woman', 'man']
print('Categories', df['Sex_cat'].cat.categories, sep='\n', end='\n\n\n')

df['Sex_cat'] = df['Sex_cat'].cat.add_categories(['Nothing'])
print('Categories', df['Sex_cat'].cat.categories, sep='\n', end='\n\n\n')
print('Codes', df['Sex_cat'].cat.codes, sep='\n', end='\n\n\n')

df['Sex_cat'] = df['Sex_cat'].cat.remove_unused_categories()
print('Categories', df['Sex_cat'].cat.categories, sep='\n', end='\n\n\n')

