"""
Phishing DATASET
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

phishing = arff.loadarff('Datasets/phishing.arff')
data= pd.DataFrame(phishing[0])

data.head()

data.info()

valores_unicos = data['Result'].unique()
print("Valores únicos na coluna 'Result':", valores_unicos)

data1 = data.drop_duplicates()

data1.shape

data1.keys()

# Converter os valores das colunas para tipo numérico
df_numeric = data1.apply(lambda x: pd.to_numeric(x, errors='coerce'))

z_scores = stats.zscore(df_numeric)

threshold = 3

outlier_indices = (abs(z_scores) > threshold).any(axis=1)

df_no_outliers = data1[~outlier_indices]

df_no_outliers.head()

data_copy = df_no_outliers.copy()

for column in data_copy.columns:
    data_copy[column] = data_copy[column].astype('category')

onehot_encoder = OneHotEncoder()

data_encoded = onehot_encoder.fit_transform(data_copy)

data_encoded_df = pd.DataFrame(data_encoded.toarray(), columns=onehot_encoder.get_feature_names_out(data_copy.columns))

data_encoded_df.head()

scaler = StandardScaler()

data_scaled = scaler.fit_transform(data_encoded_df)

data_scaled_df = pd.DataFrame(data_scaled, columns=data_encoded_df.columns)

data_scaled_df.head()

X = data_scaled_df

pca = PCA(n_components=0.90)

pca.fit(X)

n_components = pca.n_components_
print(f'Número de componentes mantidos: {n_components}')

X_pca = pca.transform(X)

X_pca_df = pd.DataFrame(X_pca, columns=[f'Component_{i}' for i in range(X_pca.shape[1])])

X = X_pca_df
y = data_scaled_df['Result_b\'-1\'']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)