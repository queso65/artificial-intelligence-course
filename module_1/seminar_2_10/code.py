import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

train = pd.read_csv("train_oil.csv")
test = pd.read_csv("oil_test.csv")
print(f"Train dataset shape: {train.shape}")
print(f"Test dataset shape: {test.shape}")
print(train.head())

print(train.isna().sum())
print(test.isna().sum())

# plt.figure(figsize=(15,7))
# cmap = sns.cubehelix_palette(as_cmap=True, light=.9)
# sns.heatmap(train.isna().transpose(), cmap=cmap,
#             cbar_kws={'label': 'Missing Data'}, linewidths=0.05)
# plt.savefig("visualizing_missing_data_in_train_with_heatmap_Seaborn_Python.png", dpi=100)
#
# plt.figure(figsize=(15,7))
# cmap = sns.cubehelix_palette(as_cmap=True, light=.9)
# sns.heatmap(test.isna().transpose(), cmap=cmap,
#             cbar_kws={'label': 'Missing Data'}, linewidths=0.05)
# plt.savefig("visualizing_missing_data_in_test_with_heatmap_Seaborn_Python.png", dpi=100)

# print(train.dtypes)
train = train.dropna()

# sns.catplot(x="Onshore/Offshore", y="Depth", kind="box", data=train)
# plt.savefig("distribution of depth by location type.png", dpi=100)
# plt.show()

print(train['Tectonic regime'].value_counts())
print(train['Structural setting'].value_counts())
print(train['Basin name'].value_counts())

train = train.drop(columns=['Tectonic regime', 'Structural setting', 'Field name', 'Reservoir unit', 'Country',
                            'Region', 'Basin name', 'Operator company'])
test = test.drop(columns=['Tectonic regime', 'Structural setting', 'Field name', 'Reservoir unit', 'Country',
                            'Region', 'Basin name', 'Operator company'])

# print(train['Hydrocarbon type'].value_counts())
# print(train['Field name'].value_counts())
# print(train['Field name'].value_counts())

# numeric_cols = train.select_dtypes(include=['number']).columns
# sns.heatmap(train[numeric_cols].corr(),
#             annot=True,
#             cmap="YlGnBu",
#             linecolor='white',
#             linewidths=1)
# plt.savefig("heatmap.png")
# plt.show()

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
fig, ax = plt.subplots(1,2, figsize=(15, 7))
sns.countplot(train['Hydrocarbon type'], ax=ax[0], palette='hls')
sns.countplot(train['Onshore/Offshore'], ax=ax[1], palette='hls')
fig.savefig("heatmap.png")
fig.show()

# Список категориальных столбцов
categ = ['Hydrocarbon type', 'Reservoir status', 'Reservoir period', 'Lithology']
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_encoded = encoder.fit_transform(train_X[categ])
train_encoded = encoder.transform(test_X[categ])

train_encoded = encoder.fit_transform(train[categ])
test_encoded = encoder.transform(test[categ])

# Заменяем старые категориальные столбцы новыми закодированными
train[categ] = train_encoded
test[categ] = test_encoded


