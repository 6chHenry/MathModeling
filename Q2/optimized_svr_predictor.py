import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import joblib
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class OptimizedSVRPredictor:
    """
    优化的Support Vector Regression电影评分预测器
    
    基于模型比较结果，SVR在准确性方面表现最佳，因此选择其作为最终模型。
    SVR的优势：
    1. 强大的非线性建模能力（通过RBF核）
    2. 对高维数据有效
    3. 泛化能力强
    4. 对异常值具有鲁棒性（ε-不敏感损失）
    """
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = []
        self.training_history = {}
        self.best_params = {}
        
    def load_and_prepare_data(self):
        """加载并准备数据"""
        print("正在加载和预处理数据...")
        
        # 加载测试数据
        test_df = pd.read_csv('input_data/df_movies_test.csv')
        print(f"测试数据形状: {test_df.shape}")
        
        # 尝试加载训练数据
        try:
            train_df = pd.read_csv('df_movies_cleaned.csv')
            print(f"训练数据形状: {train_df.shape}")
            print(f"评分范围: {train_df['rating'].min():.2f} - {train_df['rating'].max():.2f}")
        except Exception as e:
            print(f"无法加载训练数据: {e}")
            print("使用模拟数据进行演示...")
            # 创建模拟数据
            train_df = test_df.copy()
            np.random.seed(42)
            train_df['rating'] = np.random.normal(7.0, 1.5, len(train_df))
            train_df['rating'] = np.clip(train_df['rating'], 1, 10)
            # 扩展数据集
            train_df = pd.concat([train_df] * 200, ignore_index=True)
            # 添加一些真实性变化
            train_df['rating'] = np.random.normal(7.0, 1.5, len(train_df))
            train_df['rating'] = np.clip(train_df['rating'], 1, 10)
            print(f"模拟训练数据形状: {train_df.shape}")
        
        return train_df, test_df
    
    def comprehensive_feature_engineering(self, df, is_train=True):
        """
        综合特征工程
        基于电影评分的影响因素进行深度特征提取
        """
        df = df.copy()
        print(f"特征工程 - 输入数据形状: {df.shape}")
        
        # 1. 处理类型特征
        if 'genres' in df.columns:
            df['genres'] = df['genres'].fillna('Unknown')
            df['main_genre'] = df['genres'].str.split(',').str[0]
            df['genre_count'] = df['genres'].str.count(',') + 1
            
            # 主要类型特征
            popular_genres = ['Action', 'Drama', 'Comedy', 'Thriller', 'Romance', 'Horror', 'Sci-Fi', 'Fantasy']
            for genre in popular_genres:
                df[f'is_{genre.lower()}'] = df['genres'].str.contains(genre, na=False).astype(int)
        
        # 2. 处理演员特征
        if 'cast' in df.columns:
            df['cast'] = df['cast'].fillna('')
            df['cast_count'] = df['cast'].str.count(',') + 1
            df['cast_count'] = df['cast_count'].where(df['cast'] != '', 0)
            
            # 演员多样性指标
            df['has_cast'] = (df['cast'] != '').astype(int)
        
        # 3. 处理导演特征
        if 'director' in df.columns:
            df['director'] = df['director'].fillna('Unknown')
            df['has_director'] = (df['director'] != 'Unknown').astype(int)
        
        # 4. 处理编剧特征
        if 'writers' in df.columns:
            df['writers'] = df['writers'].fillna('')
            df['writers_count'] = df['writers'].str.count(',') + 1
            df['writers_count'] = df['writers_count'].where(df['writers'] != '', 0)
            df['has_writers'] = (df['writers'] != '').astype(int)
        
        # 5. 处理制片公司特征
        if 'production_companies' in df.columns:
            df['production_companies'] = df['production_companies'].fillna('')
            df['production_count'] = df['production_companies'].str.count(',') + 1
            df['production_count'] = df['production_count'].where(df['production_companies'] != '', 0)
            df['has_production'] = (df['production_companies'] != '').astype(int)
        
        # 6. 处理制片人特征
        if 'producers' in df.columns:
            df['producers'] = df['producers'].fillna('')
            df['producers_count'] = df['producers'].str.count(',') + 1
            df['producers_count'] = df['producers_count'].where(df['producers'] != '', 0)
            df['has_producers'] = (df['producers'] != '').astype(int)
        
        # 7. 处理语言特征
        if 'original_language' in df.columns:
            df['original_language'] = df['original_language'].fillna('Unknown')
            df['is_english'] = (df['original_language'].str.lower() == 'en').astype(int)
            df['is_major_language'] = df['original_language'].isin(['en', 'es', 'fr', 'de', 'it', 'ja', 'ko', 'zh']).astype(int)
        
        # 8. 处理时长特征
        if 'runtime' in df.columns:
            df['runtime'] = df['runtime'].fillna(df['runtime'].median() if is_train else 120)
            
            # 时长分类
            df['runtime_short'] = (df['runtime'] <= 90).astype(int)
            df['runtime_normal'] = ((df['runtime'] > 90) & (df['runtime'] <= 150)).astype(int)
            df['runtime_long'] = (df['runtime'] > 150).astype(int)
            
            # 时长变换
            df['runtime_log'] = np.log1p(df['runtime'])
            df['runtime_sqrt'] = np.sqrt(df['runtime'])
        
        # 9. 交互特征
        if 'cast_count' in df.columns and 'runtime' in df.columns:
            df['cast_per_minute'] = df['cast_count'] / (df['runtime'] + 1)
            df['cast_density'] = df['cast_count'] / np.log1p(df['runtime'])
        
        if 'writers_count' in df.columns and 'production_count' in df.columns:
            df['creative_team_size'] = df['writers_count'] + df['production_count']
            df['production_complexity'] = df['writers_count'] * df['production_count']
        
        # 10. 综合评分指标
        numeric_features = ['genre_count', 'cast_count', 'writers_count', 'production_count', 'producers_count']
        existing_features = [col for col in numeric_features if col in df.columns]
        if existing_features:
            df['total_people_involved'] = df[existing_features].sum(axis=1)
            df['production_scale'] = df[existing_features].mean(axis=1)
        
        print(f"特征工程后数据形状: {df.shape}")
        return df
    
    def prepare_features(self, df, fit_preprocessor=False):
        """准备特征矩阵"""
        # 数值特征
        numeric_features = [
            'runtime', 'runtime_log', 'runtime_sqrt',
            'cast_count', 'writers_count', 'production_count', 'producers_count',
            'genre_count', 'cast_per_minute', 'cast_density',
            'creative_team_size', 'production_complexity',
            'total_people_involved', 'production_scale'
        ]
        
        # 二进制特征
        binary_features = [
            'has_director', 'has_cast', 'has_writers', 'has_production', 'has_producers',
            'runtime_short', 'runtime_normal', 'runtime_long',
            'is_english', 'is_major_language'
        ]
        
        # 类型特征（二进制）
        genre_features = [
            'is_action', 'is_drama', 'is_comedy', 'is_thriller', 
            'is_romance', 'is_horror', 'is_sci-fi', 'is_fantasy'
        ]
        
        # 类别特征
        categorical_features = ['main_genre', 'original_language']
        
        # 确保所有特征都存在
        all_numeric_binary = numeric_features + binary_features + genre_features
        for col in all_numeric_binary:
            if col not in df.columns:
                df[col] = 0
        
        for col in categorical_features:
            if col not in df.columns:
                df[col] = 'Unknown'
        
        # 记录特征名称
        if fit_preprocessor:
            self.feature_names = all_numeric_binary + categorical_features
        
        # 创建预处理器
        if fit_preprocessor or self.preprocessor is None:
            self.preprocessor = ColumnTransformer([
                ('num_binary', StandardScaler(), all_numeric_binary),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ])
            X = self.preprocessor.fit_transform(df[all_numeric_binary + categorical_features])
        else:
            X = self.preprocessor.transform(df[all_numeric_binary + categorical_features])
        
        return X
    
    def optimize_hyperparameters(self, X_train, y_train, cv_folds=5):
        """
        超参数优化
        SVR的关键超参数：
        - C: 正则化参数，控制对误差的容忍度
        - gamma: RBF核参数，控制单个训练样本的影响范围
        - epsilon: ε-不敏感损失的参数
        """
        print("开始超参数优化...")
        
        # 定义参数网格
        param_grid = {
            'C': [0.1, 1, 10, 100],  # 正则化参数
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # RBF核参数
            'epsilon': [0.01, 0.1, 0.2, 0.5]  # ε-不敏感参数
        }
        
        # 创建SVR模型
        svr = SVR(kernel='rbf')
        
        # 网格搜索
        grid_search = GridSearchCV(
            svr, 
            param_grid, 
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        print("执行网格搜索...")
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        optimization_time = time.time() - start_time
        
        self.best_params = grid_search.best_params_
        print(f"超参数优化完成，用时 {optimization_time:.2f} 秒")
        print(f"最佳参数: {self.best_params}")
        print(f"最佳交叉验证分数 (负MSE): {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_model(self, X_train, y_train, optimize_hyperparams=True):
        """训练SVR模型"""
        print("开始训练SVR模型...")
        
        if optimize_hyperparams:
            # 超参数优化
            self.model = self.optimize_hyperparameters(X_train, y_train)
        else:
            # 使用默认参数或之前找到的最佳参数
            params = self.best_params if self.best_params else {'C': 1.0, 'gamma': 'scale', 'epsilon': 0.1}
            self.model = SVR(kernel='rbf', **params)
            
            print("训练模型...")
            start_time = time.time()
            self.model.fit(X_train, y_train)
            training_time = time.time() - start_time
            print(f"训练完成，用时 {training_time:.2f} 秒")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test, X_train=None, y_train=None):
        """全面评估模型"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        print("\n" + "="*60)
        print("Support Vector Regression 模型评估报告")
        print("="*60)
        
        # 预测
        y_pred = self.model.predict(X_test)
        
        # 计算指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        # 打印结果
        print(f"测试集性能:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape*100:.2f}%")
        
        # 如果有训练集，也计算训练集性能
        if X_train is not None and y_train is not None:
            y_train_pred = self.model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            print(f"\n训练集性能:")
            print(f"  RMSE: {train_rmse:.4f}")
            print(f"  R²:   {train_r2:.4f}")
            
            # 过拟合检查
            overfitting_ratio = train_rmse / rmse
            if overfitting_ratio < 0.8:
                print(f"  过拟合检查: 可能存在过拟合 (训练RMSE/测试RMSE = {overfitting_ratio:.3f})")
            else:
                print(f"  过拟合检查: 正常 (训练RMSE/测试RMSE = {overfitting_ratio:.3f})")
        
        # 交叉验证
        if X_train is not None and y_train is not None:
            cv_scores = cross_val_score(self.model, X_train, y_train, 
                                      cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            cv_rmse = np.sqrt(-cv_scores)
            print(f"\n5折交叉验证:")
            print(f"  平均RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
            print(f"  RMSE范围: {cv_rmse.min():.4f} - {cv_rmse.max():.4f}")
        
        # SVR特有信息
        print(f"\nSVR模型信息:")
        print(f"  支持向量数量: {len(self.model.support_)}")
        print(f"  支持向量比例: {len(self.model.support_)/len(y_train)*100:.1f}%" if y_train is not None else "")
        print(f"  使用的核函数: {self.model.kernel}")
        print(f"  超参数: C={self.model.C}, gamma={self.model.gamma}, epsilon={self.model.epsilon}")
        
        return {
            'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape,
            'predictions': y_pred
        }
    
    def visualize_results(self, X_test, y_test, X_train=None, y_train=None):
        """可视化结果"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        y_pred = self.model.predict(X_test)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Support Vector Regression 模型分析', fontsize=16, fontweight='bold')
        
        # 1. 预测vs实际
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue', s=30)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('实际评分')
        axes[0, 0].set_ylabel('预测评分')
        axes[0, 0].set_title('预测值 vs 实际值')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加R²信息
        r2 = r2_score(y_test, y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. 残差图
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green', s=30)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('预测评分')
        axes[0, 1].set_ylabel('残差')
        axes[0, 1].set_title('残差分析')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 误差分布
        errors = np.abs(residuals)
        axes[0, 2].hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].axvline(errors.mean(), color='red', linestyle='--', label=f'平均误差: {errors.mean():.3f}')
        axes[0, 2].set_xlabel('绝对误差')
        axes[0, 2].set_ylabel('频率')
        axes[0, 2].set_title('误差分布')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 支持向量可视化（如果特征维度允许）
        axes[1, 0].text(0.5, 0.5, f'支持向量数量: {len(self.model.support_)}\n'
                       f'总样本数: {len(y_train) if y_train is not None else "N/A"}\n'
                       f'支持向量比例: {len(self.model.support_)/len(y_train)*100:.1f}%' if y_train is not None else 'N/A',
                       ha='center', va='center', transform=axes[1, 0].transAxes,
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 0].set_title('支持向量信息')
        axes[1, 0].axis('off')
        
        # 5. 预测区间分析
        pred_ranges = [(1, 3), (3, 5), (5, 7), (7, 9), (9, 10)]
        range_labels = ['1-3', '3-5', '5-7', '7-9', '9-10']
        range_counts = []
        range_accuracy = []
        
        for low, high in pred_ranges:
            mask = (y_test >= low) & (y_test < high)
            count = mask.sum()
            if count > 0:
                accuracy = np.mean(np.abs(y_test[mask] - y_pred[mask]) < 0.5)
            else:
                accuracy = 0
            range_counts.append(count)
            range_accuracy.append(accuracy)
        
        x_pos = np.arange(len(range_labels))
        bars = axes[1, 1].bar(x_pos, range_accuracy, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('评分区间')
        axes[1, 1].set_ylabel('准确率 (±0.5)')
        axes[1, 1].set_title('各评分区间预测准确率')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(range_labels)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, count in zip(bars, range_counts):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}\n(n={count})', ha='center', va='bottom', fontsize=8)
        
        # 6. 超参数信息
        param_text = f"SVR超参数设置:\n\n"
        param_text += f"C (正则化): {self.model.C}\n"
        param_text += f"gamma (核参数): {self.model.gamma}\n"
        param_text += f"epsilon (ε): {self.model.epsilon}\n"
        param_text += f"kernel: {self.model.kernel}\n\n"
        param_text += f"模型性能:\n"
        param_text += f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}\n"
        param_text += f"MAE: {mean_absolute_error(y_test, y_pred):.4f}\n"
        param_text += f"R²: {r2:.4f}"
        
        axes[1, 2].text(0.05, 0.95, param_text, transform=axes[1, 2].transAxes,
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        axes[1, 2].set_title('模型参数与性能')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('svr_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("可视化分析完成，图片已保存为 svr_model_analysis.png")
    
    def predict_test_data(self, test_df):
        """预测测试数据"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        print("对测试数据进行预测...")
        
        # 特征工程
        test_processed = self.comprehensive_feature_engineering(test_df, is_train=False)
        X_test = self.prepare_features(test_processed, fit_preprocessor=False)
        
        # 预测
        predictions = self.model.predict(X_test)
        
        # 确保预测值在合理范围内
        predictions = np.clip(predictions, 1, 10)
        
        return predictions
    
    def save_model(self, filename='svr_movie_predictor.pkl'):
        """保存训练好的模型"""
        if self.model is None:
            raise ValueError("没有训练好的模型可以保存")
        
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'best_params': self.best_params
        }
        
        joblib.dump(model_data, filename)
        print(f"模型已保存到: {filename}")
    
    def load_model(self, filename='svr_movie_predictor.pkl'):
        """加载训练好的模型"""
        try:
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.preprocessor = model_data['preprocessor']
            self.feature_names = model_data['feature_names']
            self.best_params = model_data['best_params']
            print(f"模型已从 {filename} 加载")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False

def main():
    """主函数"""
    print("="*80)
    print("Support Vector Regression 电影评分预测系统")
    print("="*80)
    
    # 创建预测器
    predictor = OptimizedSVRPredictor()
    
    # 加载数据
    train_df, test_df = predictor.load_and_prepare_data()
    
    # 特征工程
    print("\n进行特征工程...")
    train_processed = predictor.comprehensive_feature_engineering(train_df, is_train=True)
    X = predictor.prepare_features(train_processed, fit_preprocessor=True)
    
    if 'rating' in train_processed.columns:
        y = train_processed['rating'].values
    else:
        print("未找到评分列，使用模拟数据")
        y = np.random.normal(7.0, 1.5, X.shape[0])
        y = np.clip(y, 1, 10)
    
    # 划分训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"\n数据划分:")
    print(f"  训练集大小: {X_train.shape}")
    print(f"  测试集大小: {X_test.shape}")
    print(f"  特征数量: {X_train.shape[1]}")
    
    # 训练模型
    print(f"\n开始训练SVR模型...")
    predictor.train_model(X_train, y_train, optimize_hyperparams=True)
    
    # 评估模型
    print(f"\n评估模型性能...")
    evaluation_results = predictor.evaluate_model(X_test, y_test, X_train, y_train)
    
    # 可视化结果
    print(f"\n生成可视化分析...")
    predictor.visualize_results(X_test, y_test, X_train, y_train)
    
    # 保存模型
    predictor.save_model()
    
    # 预测测试数据
    print(f"\n对测试集进行预测...")
    test_predictions = predictor.predict_test_data(test_df)
    
    # 保存预测结果
    result_df = pd.DataFrame({
        'id': test_df['id'] if 'id' in test_df.columns else range(len(test_df)),
        'predicted_rating': test_predictions
    })
    
    result_df.to_csv('output_result/svr_predictions.csv', index=False)
    
    print(f"\n预测完成!")
    print(f"  预测结果已保存到: output_result/svr_predictions.csv")
    print(f"  预测评分范围: {test_predictions.min():.2f} - {test_predictions.max():.2f}")
    print(f"  预测评分均值: {test_predictions.mean():.2f}")
    print(f"  预测评分标准差: {test_predictions.std():.2f}")
    
    print(f"\n模型训练和评估完成!")
    print("="*80)

if __name__ == "__main__":
    main()
