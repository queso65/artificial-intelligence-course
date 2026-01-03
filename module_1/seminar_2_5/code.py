import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
train = pd.read_csv("train_final.csv")
test = pd.read_csv("test_final.csv")

# print(f"Train dataset shape: {train.shape}")
# print(f"Train dataset shape: {test.shape}")

# исследование данных
# print(train.info())
# print(train.shape)
# print(train.isna().sum().sum())
#
# print(test.info())
# print(test.shape)
# print(test.isna().sum().sum())
#
# print(train['is_canceled'].value_counts())

# Предобработка данных
class_0 = train[train['is_canceled'] == 0]
class_1 = train[train['is_canceled'] == 1]
class_1_oversampled = class_1.sample(n=len(class_0), replace=True, random_state=42)
train = pd.concat([class_0, class_1_oversampled]).sample(frac=1, random_state=42).reset_index(drop=True)
# print(train['is_canceled'].value_counts())
# print(train.loc[:, train.dtypes == object])
train = train.drop(columns=['reservation_status_date', 'country', 'assigned_room_type', 'reserved_room_type'])
test = test.drop(columns=['reservation_status_date', 'country', 'assigned_room_type', 'reserved_room_type'])
cat_features = train.loc[:, train.dtypes == object].columns
# print(cat_features)
# print(train['hotel'].value_counts())
# print(train['market_segment'].value_counts())
# print(train['deposit_type'].value_counts())
categ = ['hotel', 'market_segment', 'deposit_type', 'arrival_date_month', 'market_segment', 'customer_type', 'meal', 'distribution_channel']
for col in categ:
    le = preprocessing.LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
# print(train.head())
# print(train.dtypes)

# Обучение модели
X = train.drop(columns=['is_canceled'])
y = train['is_canceled']
# print(f'X shape: {X.shape}')
# print(f'Y shape: {y.shape}')
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
answers_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, answers_pred)}')
print(f'Precision: {precision_score(y_test, answers_pred)}')
print(f'Recall: {recall_score(y_test, answers_pred)}')

y_pred_test = model.predict(test)
y_pred_test = pd.DataFrame(y_pred_test, columns=['is_canceled'])
y_pred_test = y_pred_test.reset_index()
y_pred_test.to_csv("predict.csv", index=False)










