import pandas as pd
import numpy as np
import joblib

# 加载预处理器和模型
MODEL_FILE = "xgb_model.pkl"
PREPROCESSOR_FILE = "preprocessor.pkl"

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

if __name__ == "__main__":
    # 1. 读取新数据
    file_path = r"C:\Users\47797\Desktop\df_movies_test.csv"  # 你的新数据文件路径
    df_new = pd.read_csv(file_path)

    # 2. 特征工程
    df_processed = feature_engineering(df_new)

    # 3. 加载预处理器并转换
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    numeric_features = ['runtime', 'cast_count', 'writers_count', 'production_count',
                        'genre_count', 'has_director', 'is_action', 'is_drama', 'is_comedy', 'is_english']
    categorical_features = ['main_genre', 'original_language', 'runtime_category']

    X_new = preprocessor.transform(df_processed[numeric_features + categorical_features])

    # 4. 加载模型并预测
    model = joblib.load(MODEL_FILE)
    y_pred = model.predict(X_new)

    # 5. 输出预测结果
    df_new["predicted_rating"] = y_pred
    print("\n===== 新数据预测结果（前19条） =====")
    print(df_new[["predicted_rating"]].head(20))

    # 6. 保存预测结果
    df_new.to_csv("predicted_movies.csv", index=False)
    print("\n预测结果已保存到 predicted_movies.csv")
