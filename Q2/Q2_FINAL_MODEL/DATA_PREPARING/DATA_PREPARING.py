# feature_preprocess.py
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def feature_engineering(df):
    df = df.copy()
    if 'genres' in df.columns:
        df['genres'] = df['genres'].fillna('Unknown')
        df['main_genre'] = df['genres'].str.split(',').str[0]
        df['genre_count'] = df['genres'].str.count(',') + 1
        df['is_action'] = df['genres'].str.contains('Action', na=False).astype(int)
        df['is_drama'] = df['genres'].str.contains('Drama', na=False).astype(int)
        df['is_comedy'] = df['genres'].str.contains('Comedy', na=False).astype(int)
    if 'cast' in df.columns:
        df['cast'] = df['cast'].fillna('')
        df['cast_count'] = df['cast'].str.count(',') + 1
        df['cast_count'] = df['cast_count'].where(df['cast'] != '', 0)
    if 'director' in df.columns:
        df['director'] = df['director'].fillna('Unknown')
        df['has_director'] = (df['director'] != 'Unknown').astype(int)
    if 'writers' in df.columns:
        df['writers'] = df['writers'].fillna('')
        df['writers_count'] = df['writers'].str.count(',') + 1
        df['writers_count'] = df['writers_count'].where(df['writers'] != '', 0)
    if 'production_companies' in df.columns:
        df['production_companies'] = df['production_companies'].fillna('')
        df['production_count'] = df['production_companies'].str.count(',') + 1
        df['production_count'] = df['production_count'].where(df['production_companies'] != '', 0)
    if 'original_language' in df.columns:
        df['original_language'] = df['original_language'].fillna('Unknown')
        df['is_english'] = (df['original_language'] == 'en').astype(int)
    if 'runtime' in df.columns:
        df['runtime'] = df['runtime'].fillna(df['runtime'].median())
        df['runtime_category'] = pd.cut(df['runtime'], bins=[0, 90, 120, 150, float('inf')],
                                        labels=['短片', '标准', '长片', '超长'])
    return df

def prepare_features(df):
    numeric_features = ['runtime', 'cast_count', 'writers_count', 'production_count',
                        'genre_count', 'has_director', 'is_action', 'is_drama', 'is_comedy', 'is_english']
    categorical_features = ['main_genre', 'original_language', 'runtime_category']

    for col in numeric_features:
        if col not in df.columns:
            df[col] = 0
    for col in categorical_features:
        if col not in df.columns:
            df[col] = 'Unknown'

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])
    X = preprocessor.fit_transform(df[numeric_features + categorical_features])
    return X, preprocessor

if __name__ == "__main__":
    # 1. 读取原始数据
    file_path = r"C:\Users\47797\Desktop\df_movies_train.csv"
    df = pd.read_csv(file_path)
    y = df['rating'].values

    # 2. 特征工程
    df_processed = feature_engineering(df)
    X, preprocessor = prepare_features(df_processed)

    # 3. 保存特征张量与标签
    np.savez("features_and_labels.npz", X=X, y=y)
    print("特征张量和标签已保存到 features_and_labels.npz")
import joblib
joblib.dump(preprocessor, "preprocessor.pkl")
print("已保存预处理器到 preprocessor.pkl")

