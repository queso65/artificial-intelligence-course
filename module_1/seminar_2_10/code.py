import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from itertools import combinations
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def frequency_of_combinations_in_column(df, col, top_n=5):
    counter = Counter()
    for value in df[col].dropna():
        value_str = str(value)
        parts = value_str.split('/')
        for r in range(1, len(parts) + 1):
            for combo in combinations(sorted(parts), r):
                combo_str = '/'.join(sorted(combo))
                counter[combo_str] += 1

    sorted_items = sorted(counter.items(), key=lambda x: (-x[1], len(x[0].split('/')), x[0]))
    for i, (combo, count) in enumerate(sorted_items[:top_n], 1):
        parts = combo.split('/')
        if len(parts) == 1:
            print(f"{combo:30} : {count:4}")
        else:
            print(f"{combo:30} : {count:4}")

train = pd.read_csv("train_oil.csv")
test = pd.read_csv("oil_test.csv")

# print(f"Train dataset shape: {train.shape}")
# print(f"Test dataset shape: {test.shape}")
# print(train.head())

# print(train['Country'].isna().sum())
# print(train['Latitude'].isna().sum())
# both_missing = ((train['Country'].isna()) & (train['Longitude'].isna())).sum()
# print(f"Строк с пропусками в обоих столбцах: {both_missing}")

plt.figure(figsize=(15,7))
cmap = sns.cubehelix_palette(as_cmap=True, light=.9)
sns.heatmap(train.isna().transpose(), cmap=cmap,
            cbar_kws={'label': 'Missing Data'}, linewidths=0.05)
plt.savefig("visualizing_missing_data_in_train", dpi=100)

plt.figure(figsize=(15,7))
cmap = sns.cubehelix_palette(as_cmap=True, light=.9)
sns.heatmap(test.isna().transpose(), cmap=cmap,
            cbar_kws={'label': 'Missing Data'}, linewidths=0.05)
plt.savefig("visualizing_missing_data_in_test.png", dpi=100)

# Заполняем пропуски
# print(train['Longitude'].isna().sum())
# print(train['Latitude'].isna().sum())
# print(train['Country'].isna().sum())
train['Country'] = train.groupby('Region')['Country'].transform(
    lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x
)
train['Longitude'] = train.groupby('Country')['Longitude'].transform(
    lambda x: x.fillna(x.mean())
)
train['Latitude'] = train.groupby('Country')['Latitude'].transform(
    lambda x: x.fillna(x.mean())
)

# print(train['Region'].isna().sum())
# print(train[train['Onshore/Offshore']=='ONSHORE-OFFSHORE'])

train = train.dropna()
# print(train['Operator company'].value_counts())
train = train.drop(columns=['Country', 'Tectonic regime','Field name', 'Reservoir unit',
                            'Region', 'Basin name', 'Operator company'])
test = test.drop(columns=['Country', 'Tectonic regime', 'Field name', 'Reservoir unit',
                            'Region', 'Basin name', 'Operator company'])

# print(train['Hydrocarbon type'].value_counts())
# print(train['Reservoir status'].value_counts())
# print(train['Reservoir period'].value_counts())
# print(train['Lithology'].value_counts())
categ = ['Structural setting', 'Hydrocarbon type', 'Reservoir status', 'Reservoir period', 'Lithology']
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train_encoded = encoder.fit_transform(train[categ])
train_encoded = encoder.transform(test[categ])
train_encoded = encoder.fit_transform(train[categ])
test_encoded = encoder.transform(test[categ])
train[categ] = train_encoded
test[categ] = test_encoded

# print(train['Onshore/Offshore'].value_counts())
le = LabelEncoder()
train['Onshore/Offshore'] = le.fit_transform(train['Onshore/Offshore'])

sns.set(rc = {'figure.figsize':(14, 14)})
sns.heatmap(train.corr(), annot = True, cmap="YlGnBu", linecolor='white',linewidths=1)
plt.savefig("heatmap.png")

# print(train.head())
X = train.drop(columns=['Onshore/Offshore'])
y = train['Onshore/Offshore']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# print(f'Train dataset size: {X_train.shape}, {y_train.shape}')
# print(f'Test dataset size: {X_test.shape}, {y_test.shape}')

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42,
    class_weight={
        0: 1.9,    # ONSHORE
        1: 1.0,    # OFFSHORE
        2: 1000.0    # ONSHORE-OFFSHORE
    }
)
rf.fit(X_train, y_train)
y_pred_tree = rf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred_tree)}')

conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)
display = ConfusionMatrixDisplay(conf_matrix_tree, display_labels=rf.classes_)
display.plot()
plt.gca().grid(False)
plt.title('Confusion Matrix: Decision Tree')
plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.savefig('confusion_matrix_tree.png', dpi=300, bbox_inches='tight')
plt.show()

y_test_pred = rf.predict(test)
label_mapping = {1: 'ONSHORE', 0: 'OFFSHORE', 2: 'ONSHORE-OFFSHORE'}
y_test_pred = [label_mapping[pred] for pred in y_test_pred]
ans_df = pd.DataFrame(y_test_pred, columns=['Onshore/Offshore'])
ans_df.reset_index(inplace=True)
ans_df.to_csv('predict.csv', index=False)
