import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class MovieDataset(Dataset):
    """电影数据集类"""
    def __init__(self, features, targets=None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]

class MovieRatingNN(nn.Module):
    """电影评分预测神经网络"""
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout_rate=0.3):
        super(MovieRatingNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x).squeeze()

class MovieRatingPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf_vectorizers = {}
        self.feature_names = []
        self.training_history = {'train_loss': [], 'val_loss': []}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
    
    def load_data(self):
        """加载数据"""
        print("正在加载数据...")
        
        # 加载测试数据
        test_df = pd.read_csv('input_data/df_movies_test.csv')
        print(f"测试数据形状: {test_df.shape}")
        
        # 加载训练数据
        try:
            train_df = pd.read_csv('df_movies_cleaned.csv')
            print(f"训练数据形状: {train_df.shape}")
        except Exception as e:
            print(f"无法读取训练数据: {e}")
            return None, None
        
        return train_df, test_df
    
    def advanced_feature_engineering(self, df, is_train=True):
        """高级特征工程"""
        df = df.copy()
        
        # 1. 处理类型特征 - 使用TF-IDF
        if 'genres' in df.columns:
            genres_clean = df['genres'].fillna('Unknown').str.replace(',', ' ')
            
            if is_train:
                self.tfidf_vectorizers['genres'] = TfidfVectorizer(
                    max_features=50, stop_words=None, lowercase=True
                )
                genres_tfidf = self.tfidf_vectorizers['genres'].fit_transform(genres_clean).toarray()
            else:
                genres_tfidf = self.tfidf_vectorizers['genres'].transform(genres_clean).toarray()
            
            # 添加TF-IDF特征
            for i in range(genres_tfidf.shape[1]):
                df[f'genre_tfidf_{i}'] = genres_tfidf[:, i]
            
            # 传统特征
            df['main_genre'] = df['genres'].str.split(',').str[0].fillna('Unknown')
            df['genre_count'] = df['genres'].str.count(',').fillna(0) + 1
        
        # 2. 处理演员特征
        if 'cast' in df.columns:
            cast_clean = df['cast'].fillna('Unknown').str.replace(',', ' ')
            
            if is_train:
                self.tfidf_vectorizers['cast'] = TfidfVectorizer(
                    max_features=100, stop_words=None, lowercase=True
                )
                cast_tfidf = self.tfidf_vectorizers['cast'].fit_transform(cast_clean).toarray()
            else:
                cast_tfidf = self.tfidf_vectorizers['cast'].transform(cast_clean).toarray()
            
            # 添加演员TF-IDF特征
            for i in range(cast_tfidf.shape[1]):
                df[f'cast_tfidf_{i}'] = cast_tfidf[:, i]
            
            df['cast_count'] = df['cast'].str.count(',').fillna(0) + 1
        
        # 3. 处理导演特征
        if 'director' in df.columns:
            df['has_director'] = df['director'].notna().astype(int)
            
            # 导演编码
            director_clean = df['director'].fillna('Unknown')
            if is_train:
                self.label_encoders['director'] = LabelEncoder()
                df['director_encoded'] = self.label_encoders['director'].fit_transform(director_clean)
            else:
                # 处理未见过的导演
                known_directors = set(self.label_encoders['director'].classes_)
                director_mapped = director_clean.apply(
                    lambda x: x if x in known_directors else 'Unknown'
                )
                df['director_encoded'] = self.label_encoders['director'].transform(director_mapped)
        
        # 4. 处理编剧特征
        if 'writers' in df.columns:
            df['writers_count'] = df['writers'].str.count(',').fillna(0) + 1
            df['has_writers'] = df['writers'].notna().astype(int)
        
        # 5. 处理制片公司特征
        if 'production_companies' in df.columns:
            df['production_count'] = df['production_companies'].str.count(',').fillna(0) + 1
            df['has_production'] = df['production_companies'].notna().astype(int)
        
        # 6. 处理语言特征
        if 'original_language' in df.columns:
            language_clean = df['original_language'].fillna('Unknown')
            if is_train:
                self.label_encoders['language'] = LabelEncoder()
                df['language_encoded'] = self.label_encoders['language'].fit_transform(language_clean)
            else:
                known_languages = set(self.label_encoders['language'].classes_)
                language_mapped = language_clean.apply(
                    lambda x: x if x in known_languages else 'Unknown'
                )
                df['language_encoded'] = self.label_encoders['language'].transform(language_mapped)
        
        # 7. 处理时长特征
        if 'runtime' in df.columns:
            df['runtime'] = df['runtime'].fillna(df['runtime'].median() if is_train else 120)
            # 时长分类
            df['runtime_short'] = (df['runtime'] <= 90).astype(int)
            df['runtime_normal'] = ((df['runtime'] > 90) & (df['runtime'] <= 150)).astype(int)
            df['runtime_long'] = (df['runtime'] > 150).astype(int)
            # 时长对数变换
            df['runtime_log'] = np.log1p(df['runtime'])
        
        # 8. 交互特征
        if 'cast_count' in df.columns and 'runtime' in df.columns:
            df['cast_runtime_ratio'] = df['cast_count'] / (df['runtime'] + 1)
        
        if 'writers_count' in df.columns and 'production_count' in df.columns:
            df['writers_production_ratio'] = df['writers_count'] / (df['production_count'] + 1)
        
        return df
    
    def prepare_features(self, df):
        """准备最终特征矩阵"""
        # 选择数值特征
        numeric_cols = []
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64'] and col not in ['id', 'rating']:
                numeric_cols.append(col)
        
        # 创建特征矩阵
        feature_matrix = df[numeric_cols].fillna(0)
        
        return feature_matrix.values, numeric_cols
    
    def create_model(self, input_dim):
        """创建神经网络模型"""
        model = MovieRatingNN(
            input_dim=input_dim,
            hidden_dims=[512, 256, 128, 64],
            dropout_rate=0.3
        ).to(self.device)
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=200, batch_size=64, lr=0.001):
        """训练模型"""
        print("开始训练神经网络模型...")
        
        # 创建数据加载器
        train_dataset = MovieDataset(X_train, y_train)
        val_dataset = MovieDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        self.model = self.create_model(X_train.shape[1])
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=10, min_lr=1e-6
        )
        
        # 早停机制
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        self.training_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_losses = []
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                
                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_losses.append(loss.item())
            
            # 验证阶段
            self.model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_features, batch_targets in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_targets)
                    val_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # 早停
            if patience_counter >= patience:
                print(f"早停在第 {epoch+1} 轮，最佳验证损失: {best_val_loss:.4f}")
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))
        print(f"训练完成，最终验证损失: {best_val_loss:.4f}")
    
    def predict(self, X):
        """预测"""
        self.model.eval()
        dataset = MovieDataset(X)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch_features in dataloader:
                if isinstance(batch_features, tuple):
                    batch_features = batch_features[0]
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions)
    
    def evaluate_model(self, X_val, y_val):
        """评估模型"""
        predictions = self.predict(X_val)
        
        mse = mean_squared_error(y_val, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        return metrics, predictions
    
    def visualize_results(self, metrics, y_val, predictions):
        """可视化结果"""
        plt.figure(figsize=(20, 15))
        
        # 1. 训练历史
        plt.subplot(3, 4, 1)
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        plt.plot(epochs, self.training_history['train_loss'], 'b-', label='训练损失', alpha=0.8)
        plt.plot(epochs, self.training_history['val_loss'], 'r-', label='验证损失', alpha=0.8)
        plt.title('训练和验证损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 损失放大图（后50%）
        plt.subplot(3, 4, 2)
        mid_point = len(epochs) // 2
        plt.plot(epochs[mid_point:], self.training_history['train_loss'][mid_point:], 'b-', label='训练损失', alpha=0.8)
        plt.plot(epochs[mid_point:], self.training_history['val_loss'][mid_point:], 'r-', label='验证损失', alpha=0.8)
        plt.title('损失曲线（后半段）')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 预测vs实际
        plt.subplot(3, 4, 3)
        plt.scatter(y_val, predictions, alpha=0.6, color='blue', s=20)
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
        plt.xlabel('实际评分')
        plt.ylabel('预测评分')
        plt.title(f'预测值 vs 实际值\nR² = {metrics["R2"]:.4f}')
        plt.grid(True, alpha=0.3)
        
        # 4. 残差图
        plt.subplot(3, 4, 4)
        residuals = y_val - predictions
        plt.scatter(predictions, residuals, alpha=0.6, color='green', s=20)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        plt.xlabel('预测评分')
        plt.ylabel('残差')
        plt.title('残差分布图')
        plt.grid(True, alpha=0.3)
        
        # 5. 残差直方图
        plt.subplot(3, 4, 5)
        plt.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('残差')
        plt.ylabel('频次')
        plt.title(f'残差分布直方图\nMAE = {metrics["MAE"]:.4f}')
        plt.grid(True, alpha=0.3)
        
        # 6. 评分分布对比
        plt.subplot(3, 4, 6)
        plt.hist(y_val, bins=30, alpha=0.7, label='实际评分', color='blue', edgecolor='black')
        plt.hist(predictions, bins=30, alpha=0.7, label='预测评分', color='red', edgecolor='black')
        plt.xlabel('评分')
        plt.ylabel('频次')
        plt.title('评分分布对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. 模型性能指标
        plt.subplot(3, 4, 7)
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        plt.bar(metric_names, metric_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.title('模型性能指标')
        plt.ylabel('数值')
        for i, v in enumerate(metric_values):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
        plt.xticks(rotation=45)
        
        # 8. 预测误差分布
        plt.subplot(3, 4, 8)
        absolute_errors = np.abs(residuals)
        plt.boxplot(absolute_errors)
        plt.ylabel('绝对误差')
        plt.title(f'预测误差箱线图\nRMSE = {metrics["RMSE"]:.4f}')
        plt.grid(True, alpha=0.3)
        
        # 9. 训练收敛分析
        plt.subplot(3, 4, 9)
        if len(self.training_history['train_loss']) > 20:
            # 计算最后20个epoch的平均损失
            recent_train = np.mean(self.training_history['train_loss'][-20:])
            recent_val = np.mean(self.training_history['val_loss'][-20:])
            
            plt.bar(['训练损失', '验证损失'], [recent_train, recent_val], 
                   color=['blue', 'red'], alpha=0.7)
            plt.title('最后20轮平均损失')
            plt.ylabel('Loss')
            for i, v in enumerate([recent_train, recent_val]):
                plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
        
        # 10. 学习率变化（如果有scheduler）
        plt.subplot(3, 4, 10)
        # 这里可以显示其他分析，比如特征重要性等
        plt.text(0.5, 0.5, f'总训练轮数: {len(self.training_history["train_loss"])}\n' +
                           f'最佳验证损失: {min(self.training_history["val_loss"]):.4f}\n' +
                           f'过拟合程度: {abs(metrics["R2"]):.4f}', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=12, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        plt.title('训练统计信息')
        plt.axis('off')
        
        # 11. 预测区间分析
        plt.subplot(3, 4, 11)
        score_ranges = ['1-3', '3-5', '5-7', '7-9', '9-10']
        actual_counts = [
            sum((y_val >= 1) & (y_val < 3)),
            sum((y_val >= 3) & (y_val < 5)),
            sum((y_val >= 5) & (y_val < 7)),
            sum((y_val >= 7) & (y_val < 9)),
            sum((y_val >= 9) & (y_val <= 10))
        ]
        pred_counts = [
            sum((predictions >= 1) & (predictions < 3)),
            sum((predictions >= 3) & (predictions < 5)),
            sum((predictions >= 5) & (predictions < 7)),
            sum((predictions >= 7) & (predictions < 9)),
            sum((predictions >= 9) & (predictions <= 10))
        ]
        
        x = np.arange(len(score_ranges))
        width = 0.35
        plt.bar(x - width/2, actual_counts, width, label='实际', alpha=0.7)
        plt.bar(x + width/2, pred_counts, width, label='预测', alpha=0.7)
        plt.xlabel('评分区间')
        plt.ylabel('数量')
        plt.title('评分区间分布对比')
        plt.xticks(x, score_ranges)
        plt.legend()
        
        # 12. 综合评估雷达图
        plt.subplot(3, 4, 12, projection='polar')
        # 标准化指标到0-1范围用于雷达图
        normalized_metrics = {
            'R²': max(0, metrics['R2']),  # R²可能为负
            '准确度': 1 - min(1, metrics['MAE'] / 10),  # MAE标准化
            '稳定性': 1 - min(1, metrics['RMSE'] / 10),  # RMSE标准化
            '一致性': 1 - min(1, np.std(residuals) / 5)  # 残差标准差标准化
        }
        
        categories = list(normalized_metrics.keys())
        values = list(normalized_metrics.values())
        values += values[:1]  # 闭合雷达图
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        plt.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
        plt.fill(angles, values, alpha=0.25, color='blue')
        plt.xticks(angles[:-1], categories)
        plt.ylim(0, 1)
        plt.title('模型综合性能雷达图')
        
        plt.tight_layout()
        plt.savefig('neural_network_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("可视化结果已保存为 neural_network_evaluation.png")
    
    def run_complete_pipeline(self):
        """运行完整的预测流程"""
        # 1. 加载数据
        train_df, test_df = self.load_data()
        if train_df is None:
            return
        
        print(f"训练数据: {train_df.shape}, 测试数据: {test_df.shape}")
        
        # 2. 特征工程
        print("进行高级特征工程...")
        train_processed = self.advanced_feature_engineering(train_df, is_train=True)
        test_processed = self.advanced_feature_engineering(test_df, is_train=False)
        
        # 3. 准备特征和目标
        X, feature_names = self.prepare_features(train_processed)
        y = train_processed['rating'].values
        
        print(f"特征维度: {X.shape}, 目标维度: {y.shape}")
        self.feature_names = feature_names
        
        # 4. 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 5. 分割数据
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=None
        )
        
        print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
        
        # 6. 训练模型
        self.train_model(X_train, y_train, X_val, y_val, epochs=200, batch_size=64, lr=0.001)
        
        # 7. 评估模型
        print("评估模型性能...")
        metrics, predictions = self.evaluate_model(X_val, y_val)
        
        print("=== 模型性能评估 ===")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        # 8. 可视化结果
        self.visualize_results(metrics, y_val, predictions)
        
        # 9. 预测测试数据
        print("对测试数据进行预测...")
        X_test, _ = self.prepare_features(test_processed)
        X_test_scaled = self.scaler.transform(X_test)
        test_predictions = self.predict(X_test_scaled)
        
        # 确保预测值在合理范围内
        test_predictions = np.clip(test_predictions, 1, 10)
        
        # 10. 保存结果
        result_df = pd.DataFrame({
            'id': test_df['id'] if 'id' in test_df.columns else range(len(test_df)),
            'predicted_rating': test_predictions
        })
        
        result_df.to_csv('output_result/neural_network_predictions.csv', index=False)
        
        print("\n预测完成！")
        print(f"预测评分范围: {test_predictions.min():.2f} - {test_predictions.max():.2f}")
        print(f"预测评分均值: {test_predictions.mean():.2f}")
        print(f"预测评分标准差: {test_predictions.std():.2f}")
        print("结果已保存到: output_result/neural_network_predictions.csv")
        
        return metrics, test_predictions

def main():
    """主函数"""
    predictor = MovieRatingPredictor()
    metrics, predictions = predictor.run_complete_pipeline()
    
    print("\n=== 神经网络预测完成 ===")
    return predictor, metrics, predictions

if __name__ == "__main__":
    predictor, metrics, predictions = main()
