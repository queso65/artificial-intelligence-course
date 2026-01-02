import pandas as pd
import seaborn as sns
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

train = pd.read_csv('train.csv')
formula_train = pd.read_csv('formula_train.csv')
test = pd.read_csv('test.csv')
formula_test = pd.read_csv('formula_test.csv')

# исследование данных
# print(train.shape)
# print(formula_train.shape)
# print(train.isna().sum().sum())
# print(formula_train.isna().sum().sum())
# print(train.info())
# print(formula_train.info())
#
# print(test.shape)
# print(formula_test.shape)
# print(test.isna().sum().sum())
# print(formula_test.isna().sum().sum())
# print(test.info())
# print(formula_test.info())

formula_train = formula_train.drop(columns=['critical_temp'])
train_full = pd.concat([train, formula_train], axis=1)
# print(f"Full Train dataset shape: {train_full.shape}")

# Удалим из данных ненужную колонку 'material'
train_full.drop(columns=['material'], inplace=True)
# print(train_full.head())

# Выделим из набора данных вектор признаков и вектор ответов
X = train_full.drop(columns=['critical_temp'])
y = train_full['critical_temp']
# print(f"Features shape: {X.shape}")
# print(f"Target shape: {y.shape}")

test_full = pd.concat([test, formula_test], axis=1)
# print(f"Full Test dataset shape: {test_full.shape}")
test_full.drop(columns=['material'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# print(f'Train dataset size: {X_train.shape}, {y_train.shape}')
# print(f'Test dataset size: {X_test.shape}, {y_test.shape}')

model = LinearRegression()
model.fit(X_train, y_train)
# print('Веса всех признаков (w1, ..., w167):', model.coef_)
# print('Свободный коэффицент уравнения w0:', model.intercept_)

features = test_full.columns
coeff_df = pd.DataFrame(model.coef_, columns=['Coefficient'])
coeff_df['features'] = features
# print(coeff_df.sort_values(by='Coefficient'))

y_pred = model.predict(X_test)
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R2 score:', r2_score(y_test, y_pred))

y_test_pred = model.predict(test_full)
predictions_df = pd.DataFrame({
    'critical_temp': y_test_pred
})
predictions_df.to_csv('predict.csv', index=False)





