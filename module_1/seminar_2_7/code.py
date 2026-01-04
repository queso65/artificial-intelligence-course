import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import silhouette_visualizer

def remove_correlated_features(df, threshold=0.8):
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns
               if any(upper_tri[column] > threshold)]
    print(f"Удаляем коррелирующие признаки: {to_drop}")
    return df.drop(columns=to_drop)

X = pd.read_csv("Physical_Activity_Monitoring_unlabeled.csv")

total_fixed = 0
fixed_by_type = {}

# print(X)
# print(X.columns)
# print(X.dtypes)

# df_corr = X.corr()
# f, ax = plt.subplots(figsize=(15, 10))
# sns.heatmap(df_corr, mask=np.zeros_like(df_corr, dtype=np.bool), cmap = "BrBG",ax=ax)
# plt.show()

# missing_values = X.isnull().sum()
# total_rows = len(X)
# print(missing_values)  # Количество пропусков
# print(total_rows)  # Общее число строк
# plt.figure(figsize=(10,10))
# sns.heatmap(X.isna().transpose(),
#             cmap="YlGnBu",
#             cbar_kws={'label': 'Missing Data'})
# plt.savefig("visualizing_missing_data_with_heatmap_Seaborn_Python.png", dpi=100)
# plt.show()
X.drop(columns=['handTemperature', 'handAcc16_1', 'handAcc16_2', 'handAcc16_3', 'handAcc16_3', 'handAcc6_1', 'handAcc6_2', 'handAcc6_3',
                'handGyro1', 'handGyro2', 'handGyro3', 'handMagne1', 'handMagne2', 'handMagne3', 'handOrientation1', 'handOrientation2',
                'handOrientation3', 'handOrientation4', 'ankleTemperature',
                'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 'ankleAcc6_1',
                'ankleAcc6_2', 'ankleAcc6_3', 'ankleGyro1', 'ankleGyro2', 'ankleGyro3',
                'ankleMagne1', 'ankleMagne2', 'ankleMagne3', 'ankleOrientation1',
                'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4'], inplace=True)

# print(X.isna().sum())
# print(X.columns)
col_missing = ['chestTemperature', 'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3', 'chestAcc6_1', 'chestAcc6_2',
               'chestAcc6_3', 'chestGyro1', 'chestGyro2', 'chestGyro3', 'chestMagne1',
               'chestMagne2', 'chestMagne3', 'chestOrientation1', 'chestOrientation2',
               'chestOrientation3', 'chestOrientation4']
for i in col_missing:
    # X[i].fillna(X[i].median(), inplace=True)
    X[i] = X[i].interpolate(method='linear')
print(X.isna().sum().sum())
X = remove_correlated_features(X)

# inertia_df = pd.DataFrame(data=[], index=range(2, 12), columns=['inertia'])
# for n_clusters in range(2, 12):
#     clusterer = KMeans(n_clusters=n_clusters,  random_state=42)
#     cluster_labels = clusterer.fit_predict(X)
#     # inertia
#     inertia_df.loc[n_clusters] = clusterer.inertia_
# inertia_df.plot()
# plt.xticks(range(2, 12, 1))
# plt.xlabel('Number of clusters')
# plt.ylabel('K-means score')
# plt.title("Elbow Method for selection of optimal K clusters")
# plt.savefig("Elbow_Method_for_selection_of_optimal_K_clusters.png", dpi=100)
# plt.show()

clusterer = KMeans(n_clusters=4, random_state=42)
cluster_labels = clusterer.fit_predict(X)
plt.figure(figsize=(12, 8))
visualizer = silhouette_visualizer(
    clusterer,
    X.sample(frac=0.1),
    colors='yellowbrick',
    title='Silhouette Plot for K=4',
    show=False
)
filename = f'silhouette_k_4.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f'Сохранен: {filename}')
plt.show()

predictions = pd.DataFrame(cluster_labels, columns=['activityID'])
label_mapping = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
predictions['activityID'] = predictions['activityID'].map(label_mapping)
predictions['Index'] = range(1, len(predictions) + 1)
predictions = predictions[['Index', 'activityID']]
predictions.to_csv("predict.csv", index=False)








