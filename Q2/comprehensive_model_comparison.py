import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time

# 机器学习库
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

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

# 尝试导入PyTorch用于神经网络
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MovieDataset(Dataset):
    """PyTorch数据集类"""
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
    """简单的神经网络"""
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
    """模型比较类"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.preprocessor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() and HAS_PYTORCH else 'cpu')
        
    def load_and_prepare_data(self):
        """加载并准备数据"""
        print("正在加载和预处理数据...")
        
        # 加载测试数据
        test_df = pd.read_csv('../input_data/df_movies_test.csv')
        print(f"测试数据形状: {test_df.shape}")
        
        # 尝试加载训练数据
        try:
            train_df = pd.read_csv('df_movies_cleaned.csv')
            print(f"训练数据形状: {train_df.shape}")
        except Exception as e:
            print(f"无法加载训练数据: {e}")
            print("使用模拟数据进行演示...")
            # 创建模拟数据
            train_df = test_df.copy()
            np.random.seed(42)
            train_df['rating'] = np.random.normal(7.0, 1.5, len(train_df))
            train_df['rating'] = np.clip(train_df['rating'], 1, 10)
            train_df = pd.concat([train_df] * 200, ignore_index=True)
            train_df['rating'] = np.random.normal(7.0, 1.5, len(train_df))
            train_df['rating'] = np.clip(train_df['rating'], 1, 10)
        
        return train_df, test_df
    
    def feature_engineering(self, df):
        """特征工程"""
        df = df.copy()
        
        # 处理类型特征
        if 'genres' in df.columns:
            df['genres'] = df['genres'].fillna('Unknown')
            df['main_genre'] = df['genres'].str.split(',').str[0]
            df['genre_count'] = df['genres'].str.count(',') + 1
            df['is_action'] = df['genres'].str.contains('Action', na=False).astype(int)
            df['is_drama'] = df['genres'].str.contains('Drama', na=False).astype(int)
            df['is_comedy'] = df['genres'].str.contains('Comedy', na=False).astype(int)
        
        # 处理演员特征
        if 'cast' in df.columns:
            df['cast'] = df['cast'].fillna('')
            df['cast_count'] = df['cast'].str.count(',') + 1
            df['cast_count'] = df['cast_count'].where(df['cast'] != '', 0)
        
        # 处理导演特征
        if 'director' in df.columns:
            df['director'] = df['director'].fillna('Unknown')
            df['has_director'] = (df['director'] != 'Unknown').astype(int)
        
        # 处理编剧特征
        if 'writers' in df.columns:
            df['writers'] = df['writers'].fillna('')
            df['writers_count'] = df['writers'].str.count(',') + 1
            df['writers_count'] = df['writers_count'].where(df['writers'] != '', 0)
        
        # 处理制片公司特征
        if 'production_companies' in df.columns:
            df['production_companies'] = df['production_companies'].fillna('')
            df['production_count'] = df['production_companies'].str.count(',') + 1
            df['production_count'] = df['production_count'].where(df['production_companies'] != '', 0)
        
        # 处理语言特征
        if 'original_language' in df.columns:
            df['original_language'] = df['original_language'].fillna('Unknown')
            df['is_english'] = (df['original_language'] == 'en').astype(int)
        
        # 处理时长特征
        if 'runtime' in df.columns:
            df['runtime'] = df['runtime'].fillna(df['runtime'].median())
            df['runtime_category'] = pd.cut(df['runtime'], bins=[0, 90, 120, 150, float('inf')],
                                          labels=['短片', '标准', '长片', '超长'])
        
        return df
    
    def prepare_features(self, df, fit_preprocessor=False):
        """准备特征矩阵"""
        # 数值特征
        numeric_features = ['runtime', 'cast_count', 'writers_count', 'production_count', 
                           'genre_count', 'has_director', 'is_action', 'is_drama', 'is_comedy', 'is_english']
        
        # 类别特征
        categorical_features = ['main_genre', 'original_language', 'runtime_category']
        
        # 确保所有特征都存在
        for col in numeric_features:
            if col not in df.columns:
                df[col] = 0
        
        for col in categorical_features:
            if col not in df.columns:
                df[col] = 'Unknown'
        
        # 创建预处理器
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
        """初始化所有模型"""
        print("初始化模型...")
        
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'Support Vector Regression': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        # 添加XGBoost
        if HAS_XGBOOST:
            self.models['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=0
            )
        
        # 添加LightGBM
        if HAS_LIGHTGBM:
            self.models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=-1
            )
        
        # 添加CatBoost
        if HAS_CATBOOST:
            self.models['CatBoost'] = cb.CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                random_seed=42,
                verbose=False
            )
    
    def train_pytorch_model(self, X_train, y_train, X_val, y_val, epochs=100):
        """训练PyTorch神经网络"""
        if not HAS_PYTORCH:
            return None, []
        
        print("训练神经网络...")
        
        # 创建数据集
        train_dataset = MovieDataset(X_train, y_train)
        val_dataset = MovieDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # 创建模型
        model = SimpleNN(X_train.shape[1]).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
        
        return model, (train_losses, val_losses)
    
    def evaluate_models(self, X_train, y_train, X_test, y_test):
        """评估所有模型"""
        print("开始模型评估...")
        
        self.results = {}
        training_times = {}
        
        for name, model in self.models.items():
            print(f"训练和评估 {name}...")
            
            # 记录训练时间
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            training_times[name] = training_time
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 计算指标
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            # 交叉验证分数
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='neg_mean_squared_error', n_jobs=-1)
            cv_rmse = np.sqrt(-cv_scores.mean())
            cv_std = np.sqrt(cv_scores.std())
            
            self.results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape,
                'CV_RMSE': cv_rmse,
                'CV_STD': cv_std,
                'Training_Time': training_time,
                'Predictions': y_pred
            }
            
            print(f"{name} - RMSE: {rmse:.4f}, R2: {r2:.4f}, Time: {training_time:.2f}s")
        
        # 如果有PyTorch，也评估神经网络
        if HAS_PYTORCH:
            print("训练和评估神经网络...")
            start_time = time.time()
            
            # 划分验证集用于神经网络训练
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
            nn_model, loss_history = self.train_pytorch_model(X_tr, y_tr, X_val, y_val)
            training_time = time.time() - start_time
            
            if nn_model is not None:
                # 预测
                nn_model.eval()
                X_test_tensor = torch.FloatTensor(X_test).to(self.device)
                with torch.no_grad():
                    y_pred = nn_model(X_test_tensor).cpu().numpy()
                
                # 计算指标
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                
                self.results['Neural Network'] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'MAPE': mape,
                    'CV_RMSE': rmse,  # 简化处理
                    'CV_STD': 0.0,
                    'Training_Time': training_time,
                    'Predictions': y_pred,
                    'Loss_History': loss_history
                }
                
                print(f"Neural Network - RMSE: {rmse:.4f}, R2: {r2:.4f}, Time: {training_time:.2f}s")
    
    def plot_comprehensive_comparison(self, X_test, y_test):
        """绘制综合比较图"""
        print("生成综合比较可视化...")
        
        # 创建大图
        plt.figure(figsize=(20, 16))
        
        # 1. 模型性能比较 - RMSE
        plt.subplot(3, 4, 1)
        models = list(self.results.keys())
        rmse_values = [self.results[m]['RMSE'] for m in models]
        plt.bar(models, rmse_values, color='skyblue', alpha=0.7)
        plt.title('RMSE 比较', fontsize=12, fontweight='bold')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        for i, v in enumerate(rmse_values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. R² 比较
        plt.subplot(3, 4, 2)
        r2_values = [self.results[m]['R2'] for m in models]
        plt.bar(models, r2_values, color='lightgreen', alpha=0.7)
        plt.title('R² Score 比较', fontsize=12, fontweight='bold')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        for i, v in enumerate(r2_values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. MAE 比较
        plt.subplot(3, 4, 3)
        mae_values = [self.results[m]['MAE'] for m in models]
        plt.bar(models, mae_values, color='orange', alpha=0.7)
        plt.title('MAE 比较', fontsize=12, fontweight='bold')
        plt.ylabel('MAE')
        plt.xticks(rotation=45)
        for i, v in enumerate(mae_values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 4. 训练时间比较
        plt.subplot(3, 4, 4)
        time_values = [self.results[m]['Training_Time'] for m in models]
        plt.bar(models, time_values, color='red', alpha=0.7)
        plt.title('训练时间比较', fontsize=12, fontweight='bold')
        plt.ylabel('时间 (秒)')
        plt.xticks(rotation=45)
        for i, v in enumerate(time_values):
            plt.text(i, v + 0.1, f'{v:.1f}s', ha='center', va='bottom', fontsize=8)
        
        # 5. 预测 vs 实际值 (最佳模型)
        best_model = min(models, key=lambda x: self.results[x]['RMSE'])
        plt.subplot(3, 4, 5)
        y_pred_best = self.results[best_model]['Predictions']
        plt.scatter(y_test, y_pred_best, alpha=0.6, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title(f'预测 vs 实际 ({best_model})', fontsize=12, fontweight='bold')
        
        # 6. 残差分析 (最佳模型)
        plt.subplot(3, 4, 6)
        residuals = y_test - y_pred_best
        plt.scatter(y_pred_best, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title(f'残差分析 ({best_model})', fontsize=12, fontweight='bold')
        
        # 7. 交叉验证RMSE比较
        plt.subplot(3, 4, 7)
        cv_rmse_values = [self.results[m]['CV_RMSE'] for m in models]
        cv_std_values = [self.results[m]['CV_STD'] for m in models]
        plt.bar(models, cv_rmse_values, yerr=cv_std_values, capsize=5, 
                      color='purple', alpha=0.7)
        plt.title('交叉验证 RMSE', fontsize=12, fontweight='bold')
        plt.ylabel('CV RMSE')
        plt.xticks(rotation=45)
        
        # 8. MAPE 比较
        plt.subplot(3, 4, 8)
        mape_values = [self.results[m]['MAPE'] * 100 for m in models]  # 转换为百分比
        plt.bar(models, mape_values, color='brown', alpha=0.7)
        plt.title('MAPE 比较', fontsize=12, fontweight='bold')
        plt.ylabel('MAPE (%)')
        plt.xticks(rotation=45)
        for i, v in enumerate(mape_values):
            plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # 9. 模型复杂度 vs 性能
        plt.subplot(3, 4, 9)
        complexity_scores = {
            'Linear Regression': 1,
            'Ridge Regression': 1,
            'Random Forest': 3,
            'Gradient Boosting': 4,
            'Support Vector Regression': 3,
            'XGBoost': 4,
            'LightGBM': 4,
            'CatBoost': 4,
            'Neural Network': 5
        }
        
        complexity_vals = [complexity_scores.get(m, 3) for m in models]
        plt.scatter(complexity_vals, r2_values, s=100, alpha=0.7, c=range(len(models)), cmap='viridis')
        for i, model in enumerate(models):
            plt.annotate(model, (complexity_vals[i], r2_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlabel('模型复杂度')
        plt.ylabel('R² Score')
        plt.title('复杂度 vs 性能', fontsize=12, fontweight='bold')
        
        # 10. 误差分布 (最佳模型)
        plt.subplot(3, 4, 10)
        errors = np.abs(y_test - y_pred_best)
        plt.hist(errors, bins=30, alpha=0.7, color='cyan')
        plt.xlabel('绝对误差')
        plt.ylabel('频率')
        plt.title(f'误差分布 ({best_model})', fontsize=12, fontweight='bold')
        
        # 11. 神经网络训练损失曲线
        plt.subplot(3, 4, 11)
        if 'Neural Network' in self.results and 'Loss_History' in self.results['Neural Network']:
            train_losses, val_losses = self.results['Neural Network']['Loss_History']
            epochs = range(1, len(train_losses) + 1)
            plt.plot(epochs, train_losses, 'b-', label='训练损失', alpha=0.7)
            plt.plot(epochs, val_losses, 'r-', label='验证损失', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('神经网络训练曲线', fontsize=12, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, '神经网络不可用', ha='center', va='center', 
                    transform=plt.gca().transAxes)
        
        # 12. 综合评分雷达图
        plt.subplot(3, 4, 12)
        
        # 标准化指标用于雷达图
        metrics = ['RMSE', 'R2', 'MAE', 'Training_Time']
        
        # 选择前5个模型
        top_models = sorted(models, key=lambda x: self.results[x]['RMSE'])[:5]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        ax = plt.subplot(3, 4, 12, projection='polar')
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_models)))
        
        for i, model in enumerate(top_models):
            values = []
            # RMSE (越小越好，取倒数)
            values.append(1 / (self.results[model]['RMSE'] + 0.1))
            # R2 (越大越好)
            values.append(max(0, self.results[model]['R2']))
            # MAE (越小越好，取倒数)
            values.append(1 / (self.results[model]['MAE'] + 0.1))
            # 训练时间 (越小越好，取倒数)
            values.append(1 / (self.results[model]['Training_Time'] + 0.1))
            
            values += values[:1]  # 闭合
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['RMSE↑', 'R2', 'MAE↑', 'Time↑'])
        ax.set_title('模型综合性能雷达图', fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        plt.savefig('comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("可视化完成，图片已保存为 comprehensive_model_comparison.png")
    
    def print_detailed_results(self):
        """打印详细结果"""
        print("\n" + "="*80)
        print("详细模型比较结果")
        print("="*80)
        
        # 按RMSE排序
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['RMSE'])
        
        print(f"{'模型名称':<20} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'MAPE':<8} {'训练时间':<10}")
        print("-" * 80)
        
        for model_name, metrics in sorted_models:
            print(f"{model_name:<20} {metrics['RMSE']:<8.4f} {metrics['MAE']:<8.4f} "
                  f"{metrics['R2']:<8.4f} {metrics['MAPE']*100:<8.1f}% {metrics['Training_Time']:<10.2f}s")
        
        print("\n最佳模型排名:")
        print("1. 按RMSE:", sorted_models[0][0])
        print("2. 按R²:", max(self.results.items(), key=lambda x: x[1]['R2'])[0])
        print("3. 按训练速度:", min(self.results.items(), key=lambda x: x[1]['Training_Time'])[0])
    
    def generate_summary_report(self):
        """生成总结报告"""
        report = []
        report.append("# 电影评分预测模型比较报告\n")
        
        report.append("## 模型概述\n")
        report.append("本次比较包含以下机器学习方法：\n")
        
        model_descriptions = {
            'Linear Regression': '线性回归 - 最简单的线性模型',
            'Ridge Regression': '岭回归 - 带L2正则化的线性回归',
            'Random Forest': '随机森林 - 基于决策树的集成方法',
            'Gradient Boosting': '梯度提升 - 串行的树模型集成',
            'Support Vector Regression': '支持向量回归 - 基于核函数的非线性模型',
            'XGBoost': 'XGBoost - 优化的梯度提升框架',
            'LightGBM': 'LightGBM - 微软的高效梯度提升',
            'CatBoost': 'CatBoost - Yandex的梯度提升算法',
            'Neural Network': '神经网络 - 深度学习方法'
        }
        
        for model in self.results.keys():
            if model in model_descriptions:
                report.append(f"- **{model}**: {model_descriptions[model]}")
        
        report.append("\n## 性能比较\n")
        
        # 最佳模型
        best_rmse = min(self.results.items(), key=lambda x: x[1]['RMSE'])
        best_r2 = max(self.results.items(), key=lambda x: x[1]['R2'])
        fastest = min(self.results.items(), key=lambda x: x[1]['Training_Time'])
        
        report.append(f"- **准确性最佳**: {best_rmse[0]} (RMSE: {best_rmse[1]['RMSE']:.4f})")
        report.append(f"- **拟合最佳**: {best_r2[0]} (R²: {best_r2[1]['R2']:.4f})")
        report.append(f"- **速度最快**: {fastest[0]} (时间: {fastest[1]['Training_Time']:.2f}秒)")
        
        report.append("\n## 建议\n")
        
        if best_rmse[1]['RMSE'] < 0.5:
            report.append("- 所有模型表现良好，建议选择训练速度最快的模型")
        elif best_rmse[1]['RMSE'] < 1.0:
            report.append("- 模型表现中等，建议选择准确性最佳的模型")
        else:
            report.append("- 模型表现有待提升，建议进行更多特征工程或使用深度学习")
        
        # 保存报告
        with open('model_comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("详细报告已保存为 model_comparison_report.txt")

def main():
    """主函数"""
    print("开始电影评分预测模型综合比较...")
    
    # 创建比较器
    comparator = ModelComparison()
    
    # 加载数据
    train_df, test_df = comparator.load_and_prepare_data()
    
    # 特征工程
    print("进行特征工程...")
    train_processed = comparator.feature_engineering(train_df)
    # 如果需要预测测试集，可以在这里处理测试集特征
    # test_processed = comparator.feature_engineering(test_df)
    
    # 准备特征
    X = comparator.prepare_features(train_processed, fit_preprocessor=True)
    if 'rating' in train_processed.columns:
        y = train_processed['rating'].values
    else:
        print("未找到评分列，使用模拟数据")
        y = np.random.normal(7.0, 1.5, X.shape[0])
        y = np.clip(y, 1, 10)
    
    # 划分训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 初始化模型
    comparator.initialize_models()
    
    # 评估所有模型
    comparator.evaluate_models(X_train, y_train, X_test, y_test)
    
    # 生成可视化比较
    comparator.plot_comprehensive_comparison(X_test, y_test)
    
    # 打印详细结果
    comparator.print_detailed_results()
    
    # 生成总结报告
    comparator.generate_summary_report()
    
    print("\n模型比较完成！")

if __name__ == "__main__":
    main()
