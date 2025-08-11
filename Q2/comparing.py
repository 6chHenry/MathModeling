# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer

# 尝试导入XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    print("Warning: XGBoost not installed. Will skip XGBoost comparison.")
    HAS_XGBOOST = False

# 尝试导入LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    print("Warning: LightGBM not installed. Will skip LightGBM comparison.")
    HAS_LIGHTGBM = False

# 尝试导入CatBoost
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    print("Warning: CatBoost not installed. Will skip CatBoost comparison.")
    HAS_CATBOOST = False

# 尝试导入PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_PYTORCH = True
except ImportError:
    print("Warning: PyTorch not installed. Will skip Neural Network comparison.")
    HAS_PYTORCH = False

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MovieDataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.network(x).squeeze()


class ModelComparison:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.preprocessor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() and HAS_PYTORCH else 'cpu')

    def load_and_prepare_data(self):
        print("正在加载和预处理数据...")
        test_df = pd.read_csv(r"F:\MathModeling\input_data\df_movies_train.csv")
        print(f"数据形状: {test_df.shape}")
        return test_df, test_df  # 这里用同一份做演示

    def feature_engineering(self, df):
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

    def prepare_features(self, df, fit_preprocessor=False):
        numeric_features = ['runtime', 'cast_count', 'writers_count', 'production_count',
                            'genre_count', 'has_director', 'is_action', 'is_drama', 'is_comedy', 'is_english']
        categorical_features = ['main_genre', 'original_language', 'runtime_category']
        for col in numeric_features:
            if col not in df.columns:
                df[col] = 0
        for col in categorical_features:
            if col not in df.columns:
                df[col] = 'Unknown'
        if fit_preprocessor or self.preprocessor is None:
            self.preprocessor = ColumnTransformer([
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ])
            X = self.preprocessor.fit_transform(df[numeric_features + categorical_features])
        else:
            X = self.preprocessor.transform(df[numeric_features + categorical_features])
        return X

    def initialize_models(self):
        print("初始化模型...")
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
            'Support Vector Regression': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        if HAS_XGBOOST:
            self.models['XGBoost'] = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=750,
                learning_rate=0.01,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.6,
                random_state=42,
                verbosity=0
            )
        if HAS_LIGHTGBM:
            self.models['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbosity=-1)
        if HAS_CATBOOST:
            self.models['CatBoost'] = cb.CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_seed=42, verbose=False)

    def evaluate_models(self, X_train, y_train, X_test, y_test):
        print("开始模型评估...")
        self.results = {}
        for name, model in self.models.items():
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            self.results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape,
                'Training_Time': training_time,
                'Predictions': y_pred
            }
            print(f"{name} - RMSE: {rmse:.4f}, R2: {r2:.4f}, Time: {training_time:.2f}s")

    def print_detailed_results(self):
        print("\n详细结果:")
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['RMSE'])
        for model_name, metrics in sorted_models:
            print(f"{model_name:<20} RMSE: {metrics['RMSE']:.4f}  R²: {metrics['R2']:.4f}  MAE: {metrics['MAE']:.4f}")

def main():
    comparator = ModelComparison()
    train_df, _ = comparator.load_and_prepare_data()
    train_processed = comparator.feature_engineering(train_df)
    X = comparator.prepare_features(train_processed, fit_preprocessor=True)
    y = train_processed['rating'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    comparator.initialize_models()
    comparator.evaluate_models(X_train, y_train, X_test, y_test)
    comparator.print_detailed_results()

if __name__ == "__main__":
    main()
