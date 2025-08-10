import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

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

class AdvancedMovieRatingNN(nn.Module):
    """增强的电影评分预测神经网络"""
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256, 128], dropout_rate=0.3):
        super(AdvancedMovieRatingNN, self).__init__()

        layers = []
        prev_dim = input_dim

        # 输入层归一化
        layers.append(nn.BatchNorm1d(input_dim))

        # 构建隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),  # 使用LeakyReLU防止死神经元
                nn.Dropout(dropout_rate * (0.8 ** i))  # 递减的dropout
            ])
            prev_dim = hidden_dim

        # 输出层
        layers.extend([
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出0-1，后续映射到1-10
        ])

        self.network = nn.Sequential(*layers)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        output = self.network(x)
        # 将输出从[0,1]映射到[1,10]
        return output.squeeze() * 9 + 1

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class EnhancedMovieRatingPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))  # 目标值归一化
        self.label_encoders = {}
        self.tfidf_vectorizers = {}
        self.feature_names = []
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': []
        }
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
            # 修改这里：使用正确的列名
            print(f"评分范围: {train_df['rating'].min():.2f} - {train_df['rating'].max():.2f}")
            print(f"评分分布:\n{train_df['rating'].describe()}")
        except Exception as e:
            print(f"无法读取训练数据: {e}")
            return None, None

        return train_df, test_df

    def comprehensive_feature_engineering(self, df, is_train=True):
        """综合特征工程"""
        df = df.copy()
        print(f"特征工程 - 输入数据形状: {df.shape}")

        # 1. 处理类型特征
        if 'genres' in df.columns:
            df['genres'] = df['genres'].fillna('Unknown')

            # TF-IDF向量化
            genres_text = df['genres'].str.replace(',', ' ')
            if is_train:
                self.tfidf_vectorizers['genres'] = TfidfVectorizer(
                    max_features=30, stop_words=None, lowercase=True, min_df=1
                )
                genres_tfidf = self.tfidf_vectorizers['genres'].fit_transform(genres_text).toarray()
            else:
                genres_tfidf = self.tfidf_vectorizers['genres'].transform(genres_text).toarray()

            # 添加TF-IDF特征
            for i in range(genres_tfidf.shape[1]):
                df[f'genre_tfidf_{i}'] = genres_tfidf[:, i]

            # 基础特征
            df['genre_count'] = df['genres'].str.count(',') + 1
            df['is_action'] = df['genres'].str.contains('Action', na=False).astype(int)
            df['is_drama'] = df['genres'].str.contains('Drama', na=False).astype(int)
            df['is_comedy'] = df['genres'].str.contains('Comedy', na=False).astype(int)
            df['is_thriller'] = df['genres'].str.contains('Thriller', na=False).astype(int)
            df['is_romance'] = df['genres'].str.contains('Romance', na=False).astype(int)

        # 2. 处理演员特征
        if 'cast' in df.columns:
            df['cast'] = df['cast'].fillna('Unknown')

            # 演员TF-IDF
            cast_text = df['cast'].str.replace(',', ' ')
            if is_train:
                self.tfidf_vectorizers['cast'] = TfidfVectorizer(
                    max_features=50, stop_words=None, lowercase=True, min_df=1
                )
                cast_tfidf = self.tfidf_vectorizers['cast'].fit_transform(cast_text).toarray()
            else:
                cast_tfidf = self.tfidf_vectorizers['cast'].transform(cast_text).toarray()

            for i in range(cast_tfidf.shape[1]):
                df[f'cast_tfidf_{i}'] = cast_tfidf[:, i]

            df['cast_count'] = df['cast'].str.count(',') + 1

        # 3. 处理导演特征 - 修复Unknown标签错误
        if 'director' in df.columns:
            df['director'] = df['director'].fillna('Unknown')
            df['has_director'] = (df['director'] != 'Unknown').astype(int)

            if is_train:
                # 训练时拟合编码器，确保包含'Unknown'
                unique_directors = list(df['director'].unique())
                if 'Unknown' not in unique_directors:
                    unique_directors.append('Unknown')
                
                self.label_encoders['director'] = LabelEncoder()
                self.label_encoders['director'].fit(unique_directors)
                df['director_encoded'] = self.label_encoders['director'].transform(df['director'])
            else:
                # 测试时处理未见过的导演
                known_directors = set(self.label_encoders['director'].classes_)
                df['director_mapped'] = df['director'].apply(
                    lambda x: x if x in known_directors else 'Unknown'
                )
                df['director_encoded'] = self.label_encoders['director'].transform(df['director_mapped'])

        # 4. 处理编剧特征
        if 'writers' in df.columns:
            df['writers'] = df['writers'].fillna('Unknown')
            df['writers_count'] = df['writers'].str.count(',') + 1
            df['has_writers'] = (df['writers'] != 'Unknown').astype(int)

        # 5. 处理制片公司特征
        if 'production_companies' in df.columns:
            df['production_companies'] = df['production_companies'].fillna('Unknown')
            df['production_count'] = df['production_companies'].str.count(',') + 1
            df['has_production'] = (df['production_companies'] != 'Unknown').astype(int)

        # 6. 处理制片人特征
        if 'producers' in df.columns:
            df['producers'] = df['producers'].fillna('Unknown')
            df['producers_count'] = df['producers'].str.count(',') + 1
            df['has_producers'] = (df['producers'] != 'Unknown').astype(int)

        # 7. 处理语言特征
        if 'original_language' in df.columns:
            df['original_language'] = df['original_language'].fillna('Unknown')

            if is_train:
                # 训练时拟合编码器，确保包含'Unknown'
                unique_languages = list(df['original_language'].unique())
                if 'Unknown' not in unique_languages:
                    unique_languages.append('Unknown')
                
                self.label_encoders['language'] = LabelEncoder()
                self.label_encoders['language'].fit(unique_languages)
                df['language_encoded'] = self.label_encoders['language'].transform(df['original_language'])
            else:
                # 测试时处理未见过的语言
                known_languages = set(self.label_encoders['language'].classes_)
                df['language_mapped'] = df['original_language'].apply(
                    lambda x: x if x in known_languages else 'Unknown'
                )
                df['language_encoded'] = self.label_encoders['language'].transform(df['language_mapped'])

            # 语言特征
            df['is_english'] = (df['original_language'] == 'EN').astype(int)
            df['is_mandarin'] = (df['original_language'] == 'Mandarin').astype(int)

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
            df['cast_runtime_ratio'] = df['cast_count'] / (df['runtime'] + 1)
            df['cast_density'] = df['cast_count'] / np.log1p(df['runtime'])

        if 'writers_count' in df.columns and 'production_count' in df.columns:
            df['creative_team_size'] = df['writers_count'] + df['production_count']
            df['production_complexity'] = df['writers_count'] * df['production_count']

        # 10. 统计特征
        numeric_cols = ['genre_count', 'cast_count', 'writers_count', 'production_count', 'runtime']
        for col in numeric_cols:
            if col in df.columns:
                df[f'{col}_normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)

        print(f"特征工程后数据形状: {df.shape}")
        return df

    def prepare_features(self, df, is_train=True):
        """准备训练特征"""
        # 选择数值特征
        feature_cols = []

        # TF-IDF特征
        tfidf_cols = [col for col in df.columns if 'tfidf' in col]
        feature_cols.extend(tfidf_cols)

        # 数值特征
        numeric_features = [
            'genre_count', 'cast_count', 'writers_count', 'production_count',
            'runtime', 'runtime_log', 'runtime_sqrt',
            'director_encoded', 'language_encoded',
            'cast_runtime_ratio', 'cast_density', 'creative_team_size', 'production_complexity'
        ]

        # 二进制特征
        binary_features = [
            'has_director', 'has_writers', 'has_production', 'has_producers',
            'runtime_short', 'runtime_normal', 'runtime_long',
            'is_action', 'is_drama', 'is_comedy', 'is_thriller', 'is_romance',
            'is_english', 'is_mandarin'
        ]

        # 归一化特征
        normalized_features = [col for col in df.columns if col.endswith('_normalized')]

        # 合并所有特征
        all_features = numeric_features + binary_features + normalized_features
        available_features = [col for col in all_features if col in df.columns]
        feature_cols.extend(available_features)

        # 去重
        feature_cols = list(set(feature_cols))

        if is_train:
            self.feature_names = feature_cols

        print(f"使用 {len(feature_cols)} 个特征进行训练")

        # 提取特征矩阵
        X = df[feature_cols].values

        # 处理无穷大和NaN
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        return X

    def train_with_cross_validation(self, X, y, n_splits=5, epochs=200, batch_size=64, learning_rate=0.001):
        """使用交叉验证训练模型"""
        print(f"开始交叉验证训练，数据形状: X={X.shape}, y={y.shape}")

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\n=== 第 {fold+1} 折交叉验证 ===")

            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # 标准化特征
            fold_scaler = StandardScaler()
            X_train_scaled = fold_scaler.fit_transform(X_train_fold)
            X_val_scaled = fold_scaler.transform(X_val_fold)

            # 创建数据加载器
            train_dataset = MovieDataset(X_train_scaled, y_train_fold)
            val_dataset = MovieDataset(X_val_scaled, y_val_fold)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # 创建模型
            model = AdvancedMovieRatingNN(X.shape[1]).to(self.device)
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.8, patience=10
            )
            criterion = nn.MSELoss()

            # 训练模型
            best_val_loss = float('inf')
            for epoch in range(epochs):
                # 训练阶段
                model.train()
                train_losses = []
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    train_losses.append(loss.item())

                # 验证阶段
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_losses.append(loss.item())

                avg_train_loss = np.mean(train_losses)
                avg_val_loss = np.mean(val_losses)
                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss

                if epoch % 20 == 0:
                    print(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # 计算最终验证分数
            model.eval()
            with torch.no_grad():
                val_predictions = []
                for batch_X, _ in val_loader:
                    batch_X = batch_X.to(self.device)
                    outputs = model(batch_X)
                    val_predictions.extend(outputs.cpu().numpy())

            val_r2 = r2_score(y_val_fold, val_predictions)
            cv_scores.append(val_r2)
            print(f"第 {fold+1} 折 R2 分数: {val_r2:.4f}")

        print(f"\n交叉验证平均 R2 分数: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        return np.mean(cv_scores)

    def train(self, X, y, validation_split=0.2, epochs=300, batch_size=64, learning_rate=0.001):
        """训练模型"""
        print(f"开始训练，数据形状: X={X.shape}, y={y.shape}")
        print(f"目标值范围: {y.min():.2f} - {y.max():.2f}")

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=None
        )

        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        print(f"训练集大小: {X_train_scaled.shape[0]}, 验证集大小: {X_val_scaled.shape[0]}")

        # 创建数据加载器
        train_dataset = MovieDataset(X_train_scaled, y_train)
        val_dataset = MovieDataset(X_val_scaled, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 创建模型
        self.model = AdvancedMovieRatingNN(X.shape[1]).to(self.device)

        # 优化器和调度器
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=15, min_lr=1e-6
        )

        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(patience=25, min_delta=0.001)

        print("开始训练...")
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_losses = []
            train_predictions = []
            train_targets = []

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())
                train_predictions.extend(outputs.cpu().detach().numpy())
                train_targets.extend(batch_y.cpu().numpy())

            # 验证阶段
            self.model.eval()
            val_losses = []
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_losses.append(loss.item())
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())

            # 计算指标
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)

            train_r2 = r2_score(train_targets, train_predictions)
            val_r2 = r2_score(val_targets, val_predictions)

            # 记录历史
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['train_r2'].append(train_r2)
            self.training_history['val_r2'].append(val_r2)

            # 学习率调度
            scheduler.step(avg_val_loss)

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')

            # 早停检查
            early_stopping(avg_val_loss)

            # 打印进度
            if epoch % 10 == 0 or epoch < 10:
                print(f"Epoch {epoch:3d}/{epochs}: "
                      f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                      f"Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}")

            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        # 加载最佳模型
        if best_val_loss < float('inf'):
            self.model.load_state_dict(torch.load('best_model.pth'))
            print(f"训练完成，最佳验证损失: {best_val_loss:.4f}")

        return self.training_history

    def predict(self, X):
        """预测"""
        if self.model is None:
            raise ValueError("模型尚未训练")

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()

        return predictions

    def visualize_training(self):
        """可视化训练过程"""
        if not self.training_history['train_loss']:
            print("没有训练历史记录")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.training_history['train_loss']) + 1)

        # 损失曲线
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='训练损失', alpha=0.8)
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='验证损失', alpha=0.8)
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # R2分数
        ax2.plot(epochs, self.training_history['train_r2'], 'b-', label='训练R2', alpha=0.8)
        ax2.plot(epochs, self.training_history['val_r2'], 'r-', label='验证R2', alpha=0.8)
        ax2.set_title('训练和验证R2分数')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R2 Score')
        ax2.legend()
        ax2.grid(True)

        # 损失对比（对数尺度）
        ax3.semilogy(epochs, self.training_history['train_loss'], 'b-', label='训练损失')
        ax3.semilogy(epochs, self.training_history['val_loss'], 'r-', label='验证损失')
        ax3.set_title('损失曲线（对数尺度）')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Log Loss')
        ax3.legend()
        ax3.grid(True)

        # 损失差异
        loss_diff = np.array(self.training_history['val_loss']) - np.array(self.training_history['train_loss'])
        ax4.plot(epochs, loss_diff, 'g-', alpha=0.8)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('过拟合监控（验证损失 - 训练损失）')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Difference')
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig('neural_network_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate_model(self, X_test, y_test):
        """评估模型"""
        predictions = self.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print("\n模型评估结果:")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R2:   {r2:.4f}")

        # 预测vs实际值图
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(y_test, predictions, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('实际评分')
        plt.ylabel('预测评分')
        plt.title(f'预测vs实际评分 (R2={r2:.4f})')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        residuals = y_test - predictions
        plt.scatter(predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测评分')
        plt.ylabel('残差')
        plt.title('残差图')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()

        return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def main():
    """主函数"""
    # 创建预测器
    predictor = EnhancedMovieRatingPredictor()

    # 加载数据
    train_df, test_df = predictor.load_data()
    if train_df is None or test_df is None:
        print("数据加载失败")
        return

    print("开始特征工程...")

    # 训练集特征工程
    train_processed = predictor.comprehensive_feature_engineering(train_df, is_train=True)
    X_train = predictor.prepare_features(train_processed, is_train=True)
    y_train = train_processed['rating'].values  # 修改这里：使用正确的列名

    print(f"训练特征形状: {X_train.shape}")
    print(f"目标值统计: min={y_train.min():.2f}, max={y_train.max():.2f}, mean={y_train.mean():.2f}")

    # 划分训练和测试集
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # 交叉验证评估
    print("\n=== 交叉验证评估 ===")
    cv_score = predictor.train_with_cross_validation(
        X_train_split, y_train_split, n_splits=5, epochs=100
    )

    # 完整训练
    print("\n=== 完整模型训练 ===")
    history = predictor.train(
        X_train_split, y_train_split,
        validation_split=0.2,
        epochs=300,
        batch_size=32,
        learning_rate=0.001
    )

    # 可视化训练过程
    predictor.visualize_training()

    # 评估模型
    print("\n=== 模型评估 ===")
    evaluation = predictor.evaluate_model(X_test_split, y_test_split)

    # 测试集预测
    print("\n=== 测试集预测 ===")
    test_processed = predictor.comprehensive_feature_engineering(test_df, is_train=False)
    X_test_final = predictor.prepare_features(test_processed, is_train=False)

    test_predictions = predictor.predict(X_test_final)

    # 保存结果
    result_df = pd.DataFrame()
    result_df['id'] = test_df['id'] if 'id' in test_df.columns else range(len(test_predictions))
    result_df['predicted_rating'] = test_predictions
    result_df.to_csv('output_result/neural_network_predictions.csv', index=False)

    print("测试集预测完成，结果保存到 neural_network_predictions.csv")
    print(f"预测评分范围: {test_predictions.min():.2f} - {test_predictions.max():.2f}")
    print(f"预测评分均值: {test_predictions.mean():.2f}")

if __name__ == "__main__":
    main()
