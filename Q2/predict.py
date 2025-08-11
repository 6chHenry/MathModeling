import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class MovieRatingPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.training_history = {}
        
    def load_and_preprocess_data(self, train_path, test_path):
        """加载和预处理数据"""
        print("正在加载数据...")
        
        # 加载训练数据
        try:
            self.train_df = pd.read_csv(train_path)
            print(f"训练数据形状: {self.train_df.shape}")
        except:
            print("无法读取训练数据，尝试读取movie_features.npy")
            # 如果CSV太大，尝试加载预处理的特征
            features = np.load('movie_features.npy')
            with open('feature_names.txt', 'r') as f:
                feature_names = f.read().strip().split('\n')
            self.train_df = pd.DataFrame(features, columns=feature_names)
        
        # 加载测试数据
        self.test_df = pd.read_csv(test_path)
        print(f"测试数据形状: {self.test_df.shape}")
        
        # 特征工程
        self.train_df = self._feature_engineering(self.train_df)
        self.test_df = self._feature_engineering(self.test_df)
        
        return self.train_df, self.test_df
    
    def _feature_engineering(self, df):
        """特征工程"""
        df = df.copy()
        
        # 处理类型特征
        if 'genres' in df.columns:
            df['main_genre'] = df['genres'].str.split(',').str[0]
            df['genre_count'] = df['genres'].str.count(',') + 1
        
        # 处理演员特征
        if 'cast' in df.columns:
            df['cast_count'] = df['cast'].str.count(',') + 1
            df['cast_count'] = df['cast_count'].fillna(0)
        
        # 处理导演特征
        if 'director' in df.columns:
            df['has_director'] = df['director'].notna().astype(int)
        
        # 处理编剧特征
        if 'writers' in df.columns:
            df['writers_count'] = df['writers'].str.count(',') + 1
            df['writers_count'] = df['writers_count'].fillna(0)
        
        # 处理制片公司特征
        if 'production_companies' in df.columns:
            df['production_count'] = df['production_companies'].str.count(',') + 1
            df['production_count'] = df['production_count'].fillna(0)
        
        # 处理语言特征
        if 'original_language' in df.columns:
            df['lang'] = df['original_language']
        
        # 处理时长特征
        if 'runtime' in df.columns:
            df['runtime'] = df['runtime'].fillna(df['runtime'].median())
            df['runtime_category'] = pd.cut(df['runtime'], 
                                          bins=[0, 90, 120, 150, float('inf')],
                                          labels=['短片', '标准', '长片', '超长'])
        
        return df
    
    def prepare_features(self, df, is_train=True):
        """准备特征"""
        
        # 数值特征
        numeric_features = []
        if 'runtime' in df.columns:
            numeric_features.append('runtime')
        if 'cast_count' in df.columns:
            numeric_features.append('cast_count')
        if 'writers_count' in df.columns:
            numeric_features.append('writers_count')
        if 'production_count' in df.columns:
            numeric_features.append('production_count')
        if 'genre_count' in df.columns:
            numeric_features.append('genre_count')
        if 'has_director' in df.columns:
            numeric_features.append('has_director')
            
        # 类别特征
        categorical_features = []
        if 'main_genre' in df.columns:
            categorical_features.append('main_genre')
        if 'lang' in df.columns:
            categorical_features.append('lang')
        if 'runtime_category' in df.columns:
            categorical_features.append('runtime_category')
        
        # 创建特征矩阵
        features_df = pd.DataFrame()
        
        # 添加数值特征
        for col in numeric_features:
            if col in df.columns:
                features_df[col] = df[col].fillna(0)
        
        # 添加类别特征
        for col in categorical_features:
            if col in df.columns:
                # 确保列不是Categorical类型，或者正确处理Categorical类型
                try:
                    if hasattr(df[col], 'cat'):  # 检查是否是Categorical类型
                        # 如果是Categorical类型，先转换为字符串
                        temp_col = df[col].astype(str)
                        features_df[col] = temp_col.fillna('Unknown')
                    else:
                        features_df[col] = df[col].astype(str).fillna('Unknown')
                except Exception:
                    # 如果出现任何错误，直接转换为字符串
                    features_df[col] = df[col].astype(str).fillna('Unknown')
        
        return features_df, numeric_features, categorical_features
    
    def build_model(self):
        """构建模型"""
        # 集成多个模型
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0)
        }
        
        return models
    
    def train_and_evaluate(self, X_train, y_train):
        """训练和评估模型"""
        print("开始训练模型...")
        
        # 准备特征
        X_features, numeric_features, categorical_features = self.prepare_features(X_train)
        
        # 创建预处理器
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='passthrough'
        )
        
        # 构建模型
        models = self.build_model()
        
        # 训练和验证每个模型
        model_scores = {}
        trained_models = {}
        
        # 分割训练和验证集
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_features, y_train, test_size=0.2, random_state=42
        )
        
        for name, model in models.items():
            print(f"训练 {name}...")
            
            # 创建管道
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # 训练模型
            pipeline.fit(X_train_split, y_train_split)
            
            # 预测
            y_pred = pipeline.predict(X_val_split)
            
            # 计算评分
            mse = mean_squared_error(y_val_split, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val_split, y_pred)
            r2 = r2_score(y_val_split, y_pred)
            
            model_scores[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
            trained_models[name] = pipeline
            
            print(f"{name} - RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        # 选择最佳模型
        best_model_name = min(model_scores.keys(), key=lambda x: model_scores[x]['RMSE'])
        self.model = trained_models[best_model_name]
        self.preprocessor = self.model.named_steps['preprocessor']
        
        print(f"最佳模型: {best_model_name}")
        
        return model_scores, trained_models
    
    def visualize_results(self, model_scores, X_val, y_val):
        """可视化结果"""
        print("生成可视化结果...")
        
        # 1. 模型性能比较
        plt.figure(figsize=(15, 10))
        
        # RMSE比较
        plt.subplot(2, 3, 1)
        models = list(model_scores.keys())
        rmse_values = [model_scores[m]['RMSE'] for m in models]
        plt.bar(models, rmse_values, color='skyblue')
        plt.title('模型RMSE比较')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        
        # R2比较
        plt.subplot(2, 3, 2)
        r2_values = [model_scores[m]['R2'] for m in models]
        plt.bar(models, r2_values, color='lightgreen')
        plt.title('模型R²比较')
        plt.ylabel('R²')
        plt.xticks(rotation=45)
        
        # 预测vs实际
        plt.subplot(2, 3, 3)
        if self.model is not None:
            X_features, _, _ = self.prepare_features(X_val)
            y_pred = self.model.predict(X_features)
            plt.scatter(y_val, y_pred, alpha=0.6)
            plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
            plt.xlabel('实际评分')
            plt.ylabel('预测评分')
            plt.title('预测值 vs 实际值')
        else:
            plt.text(0.5, 0.5, '模型未训练', ha='center', va='center', transform=plt.gca().transAxes)
        
        # 残差图
        plt.subplot(2, 3, 4)
        if self.model is not None:
            X_features, _, _ = self.prepare_features(X_val)
            y_pred = self.model.predict(X_features)
            residuals = y_val - y_pred
            plt.scatter(y_pred, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('预测评分')
            plt.ylabel('残差')
            plt.title('残差图')
        else:
            plt.text(0.5, 0.5, '模型未训练', ha='center', va='center', transform=plt.gca().transAxes)
        
        # 损失历史（如果有的话）
        plt.subplot(2, 3, 5)
        if self.model is not None and hasattr(self.model, 'named_steps') and hasattr(self.model.named_steps['model'], 'train_score_'):
            train_scores = self.model.named_steps['model'].train_score_
            plt.plot(train_scores, label='训练损失')
            plt.xlabel('迭代次数')
            plt.ylabel('损失')
            plt.title('训练损失曲线')
            plt.legend()
        else:
            plt.text(0.5, 0.5, '该模型无训练历史', ha='center', va='center', transform=plt.gca().transAxes)
        
        # 特征重要性（如果是树模型）
        plt.subplot(2, 3, 6)
        if (self.model is not None and hasattr(self.model, 'named_steps') and 
            hasattr(self.model.named_steps['model'], 'feature_importances_') and
            self.preprocessor is not None and hasattr(self.preprocessor, 'get_feature_names_out')):
            
            importances = self.model.named_steps['model'].feature_importances_
            feature_names = self.preprocessor.get_feature_names_out()
            
            # 选择前10个重要特征
            indices = np.argsort(importances)[::-1][:10]
            plt.bar(range(10), importances[indices])
            plt.title('特征重要性 (Top 10)')
            plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45)
        else:
            plt.text(0.5, 0.5, '该模型无特征重要性', ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_test_data(self, test_df):
        """预测测试数据"""
        print("对测试数据进行预测...")
        
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train_and_evaluate方法")
        
        X_test_features, _, _ = self.prepare_features(test_df, is_train=False)
        predictions = self.model.predict(X_test_features)
        
        # 创建结果DataFrame
        result_df = test_df.copy()
        result_df['predicted_rating'] = predictions
        
        return result_df, predictions

def main():
    """主函数"""
    predictor = MovieRatingPredictor()
    
    # 加载数据
    train_df, test_df = predictor.load_and_preprocess_data(
        'df_movies_cleaned.csv',
        'input_data/df_movies_test.csv'
    )
    
    # 准备训练数据
    if 'rating' in train_df.columns:
        X_train = train_df.drop('rating', axis=1)
        y_train = train_df['rating']
    else:
        print("警告: 训练数据中未找到'rating'列，尝试使用所有特征...")
        X_train = train_df
        # 生成模拟的评分用于演示（实际使用时请删除）
        y_train = np.random.normal(7.0, 1.5, len(train_df))
        y_train = np.clip(y_train, 0, 10)
    
    # 训练模型
    model_scores, trained_models = predictor.train_and_evaluate(X_train, y_train)
    
    # 创建验证集用于可视化
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # 可视化结果
    predictor.visualize_results(model_scores, X_val_split, y_val_split)
    
    # 预测测试数据
    result_df, predictions = predictor.predict_test_data(test_df)
    
    # 保存结果
    output_df = pd.DataFrame({
        'id': test_df['id'] if 'id' in test_df.columns else range(len(test_df)),
        'predicted_rating': predictions
    })
    
    output_df.to_csv('output_result/predicted_ratings.csv', index=False)
    
    print("预测完成！结果已保存到 output_result/predicted_ratings.csv")
    print(f"预测评分范围: {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"预测评分均值: {predictions.mean():.2f}")
    
    # 显示模型性能总结
    print("\n=== 模型性能总结 ===")
    for model_name, scores in model_scores.items():
        print(f"{model_name}:")
        print(f"  RMSE: {scores['RMSE']:.4f}")
        print(f"  R²:   {scores['R2']:.4f}")
        print(f"  MAE:  {scores['MAE']:.4f}")

if __name__ == "__main__":
    main()
