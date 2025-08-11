import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# 尝试导入shap，如果不存在则提供替代方案
try:
    # import shap  # 已在顶部处理
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP库未安装，将跳过SHAP分析部分")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ModelRobustnessAnalyzer:
    """
    模型鲁棒性分析器
    
    提供全面的模型鲁棒性测试功能，包括：
    - 5折交叉验证
    - SHAP值分析
    - 特征消融研究
    - 特征扰动分析
    - 超参数邻域网格搜索
    """
    
    def __init__(self, data_path, target_column='rating'):
        """
        初始化鲁棒性分析器
        
        Args:
            data_path: 数据文件路径
            target_column: 目标变量列名
        """
        self.data_path = data_path
        self.target_column = target_column
        self.df = pd.DataFrame()  # 初始化为空DataFrame
        self.X = np.array([])  # 初始化为空数组
        self.y = np.array([])  # 初始化为空数组
        self.preprocessor = None
        self.model = None
        self.feature_names = np.array([])  # 初始化为空数组
        self.numeric_features = ['runtime', 'cast_count', 'writers_count', 'production_count',
                               'genre_count', 'has_director', 'is_action', 'is_drama',
                               'is_comedy', 'is_english']
        self.categorical_features = ['main_genre', 'original_language', 'runtime_category']
        
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("加载数据...")
        self.df = pd.read_csv(self.data_path)
        
        # 特征工程
        self.df = self._feature_engineering(self.df)
        
        # 准备特征和标签
        self.X, self.preprocessor = self._prepare_features(self.df)
        self.y = self.df[self.target_column].values
        
        # 获取特征名称
        self._get_feature_names()
        
        print(f"数据加载完成，共 {len(self.df)} 条记录，{self.X.shape[1]} 个特征")
        
    def _feature_engineering(self, df):
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
    
    def _prepare_features(self, df):
        """准备特征"""
        # 确保所有需要的特征都存在
        for col in self.numeric_features:
            if col not in df.columns:
                df[col] = 0
        for col in self.categorical_features:
            if col not in df.columns:
                df[col] = 'Unknown'
        
        # 创建预处理器
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), self.numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features)
        ])
        
        # 转换特征
        X = preprocessor.fit_transform(df[self.numeric_features + self.categorical_features])
        
        return X, preprocessor
    
    def _get_feature_names(self):
        """获取特征名称"""
        if self.preprocessor is None:
            return
            
        # 获取数值特征名称
        num_features = self.numeric_features
        
        # 获取分类特征名称
        cat_encoder = self.preprocessor.named_transformers_['cat']
        cat_features = cat_encoder.get_feature_names_out(self.categorical_features)
        
        # 合并特征名称
        self.feature_names = np.concatenate([num_features, cat_features])
        
    def train_base_model(self):
        """训练基础模型"""
        print("训练基础XGBoost模型...")
        
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=750,
            learning_rate=0.01,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.6,
            random_state=42
        )
        
        self.model.fit(self.X, self.y)
        print("基础模型训练完成")
        
    def cross_validation_analysis(self):
        """5折交叉验证分析"""
        print("执行5折交叉验证...")
        
        if self.model is None or (hasattr(self.X, '__len__') and len(self.X) == 0):
            print("错误：模型或数据未准备好")
            return None
            
        # 创建5折交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # 执行交叉验证
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=kf, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        # 计算其他指标
        cv_r2 = cross_val_score(self.model, self.X, self.y, cv=kf, scoring='r2')
        cv_mae = cross_val_score(self.model, self.X, self.y, cv=kf, scoring='neg_mean_absolute_error')
        cv_mae = -cv_mae
        
        # 保存结果
        cv_results = {
            'RMSE': cv_rmse,
            'R2': cv_r2,
            'MAE': cv_mae
        }
        
        # 打印结果
        print("交叉验证结果:")
        print(f"RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
        print(f"R2: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
        print(f"MAE: {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
        
        # 可视化交叉验证结果
        self._visualize_cv_results(cv_results)
        
        return cv_results
    
    def _visualize_cv_results(self, cv_results):
        """可视化交叉验证结果"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # RMSE分布
        axes[0].boxplot(cv_results['RMSE'])
        axes[0].set_title('RMSE分布')
        axes[0].set_ylabel('RMSE')
        axes[0].grid(True, alpha=0.3)
        
        # R2分布
        axes[1].boxplot(cv_results['R2'])
        axes[1].set_title('R²分布')
        axes[1].set_ylabel('R²')
        axes[1].grid(True, alpha=0.3)
        
        # MAE分布
        axes[2].boxplot(cv_results['MAE'])
        axes[2].set_title('MAE分布')
        axes[2].set_ylabel('MAE')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Q2/cross_validation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def shap_analysis(self):
        """SHAP值分析"""
        if not SHAP_AVAILABLE:
            print("SHAP库不可用，跳过SHAP分析")
            return None
            
        print("执行SHAP值分析...")
        
        if self.model is None or (hasattr(self.X, '__len__') and len(self.X) == 0):
            print("错误：模型或数据未准备好")
            return None
        
        try:
            # 创建SHAP解释器
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X)
            
            # 可视化SHAP值
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, self.X, feature_names=self.feature_names, show=False)
            plt.tight_layout()
            plt.savefig('Q2/shap_summary_plot.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 特征重要性条形图
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, self.X, feature_names=self.feature_names,
                             plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig('Q2/shap_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 保存SHAP值
            shap_df = pd.DataFrame(shap_values, columns=self.feature_names)
            shap_df.to_csv('Q2/shap_values.csv', index=False)
            
            print("SHAP值分析完成，结果已保存")
            
            return shap_values
        except Exception as e:
            print(f"SHAP分析过程中出错: {str(e)}")
            return None
    
    def feature_ablation_study(self):
        """特征消融研究"""
        print("执行特征消融研究...")
        
        if self.model is None or (hasattr(self.X, '__len__') and len(self.X) == 0) or (hasattr(self.feature_names, '__len__') and len(self.feature_names) == 0):
            print("错误：模型、数据或特征名称未准备好")
            return None
        
        # 基础模型性能
        base_pred = self.model.predict(self.X)
        base_rmse = np.sqrt(mean_squared_error(self.y, base_pred))
        base_r2 = r2_score(self.y, base_pred)
        
        results = []
        
        # 对每个特征进行消融测试
        for i, feature in enumerate(self.feature_names):
            # 创建消融数据（将该特征设为0）
            X_ablated = self.X.copy() if hasattr(self.X, 'copy') else self.X.toarray().copy()
            X_ablated[:, i] = 0
            
            # 预测并计算性能
            ablated_pred = self.model.predict(X_ablated)
            ablated_rmse = np.sqrt(mean_squared_error(self.y, ablated_pred))
            ablated_r2 = r2_score(self.y, ablated_pred)
            
            # 计算性能变化
            rmse_change = ablated_rmse - base_rmse
            r2_change = ablated_r2 - base_r2
            
            results.append({
                'feature': feature,
                'base_rmse': base_rmse,
                'ablated_rmse': ablated_rmse,
                'rmse_change': rmse_change,
                'base_r2': base_r2,
                'ablated_r2': ablated_r2,
                'r2_change': r2_change
            })
        
        # 转换为DataFrame并排序
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('rmse_change', ascending=False)
        
        # 可视化结果
        self._visualize_ablation_results(results_df)
        
        # 保存结果
        results_df.to_csv('Q2/feature_ablation_results.csv', index=False)
        
        print("特征消融研究完成，结果已保存")
        
        return results_df
    
    def _visualize_ablation_results(self, results_df):
        """可视化特征消融结果"""
        # 选择前15个最重要的特征
        top_features = results_df.head(15)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSE变化
        axes[0].barh(top_features['feature'], top_features['rmse_change'])
        axes[0].set_xlabel('RMSE变化')
        axes[0].set_title('特征消融对RMSE的影响')
        axes[0].grid(True, alpha=0.3)
        
        # R2变化
        axes[1].barh(top_features['feature'], top_features['r2_change'])
        axes[1].set_xlabel('R²变化')
        axes[1].set_title('特征消融对R²的影响')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Q2/feature_ablation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def feature_perturbation_analysis(self, perturbation_levels=[0.8, 0.9, 1.1, 1.2]):
        """特征扰动分析"""
        print("执行特征扰动分析...")
        
        if self.model is None or (hasattr(self.X, '__len__') and len(self.X) == 0) or (hasattr(self.feature_names, '__len__') and len(self.feature_names) == 0):
            print("错误：模型、数据或特征名称未准备好")
            return None
        
        # 基础模型性能
        base_pred = self.model.predict(self.X)
        base_rmse = np.sqrt(mean_squared_error(self.y, base_pred))
        base_r2 = r2_score(self.y, base_pred)
        
        results = []
        
        # 对每个特征进行扰动测试
        for i, feature in enumerate(self.feature_names):
            for perturbation in perturbation_levels:
                # 创建扰动数据
                X_perturbed = self.X.copy() if hasattr(self.X, 'copy') else self.X.toarray().copy()
                X_perturbed[:, i] *= perturbation
                
                # 预测并计算性能
                perturbed_pred = self.model.predict(X_perturbed)
                perturbed_rmse = np.sqrt(mean_squared_error(self.y, perturbed_pred))
                perturbed_r2 = r2_score(self.y, perturbed_pred)
                
                # 计算性能变化
                rmse_change = perturbed_rmse - base_rmse
                r2_change = perturbed_r2 - base_r2
                
                results.append({
                    'feature': feature,
                    'perturbation': perturbation,
                    'base_rmse': base_rmse,
                    'perturbed_rmse': perturbed_rmse,
                    'rmse_change': rmse_change,
                    'base_r2': base_r2,
                    'perturbed_r2': perturbed_r2,
                    'r2_change': r2_change
                })
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 可视化结果
        self._visualize_perturbation_results(results_df)
        
        # 保存结果
        results_df.to_csv('Q2/feature_perturbation_results.csv', index=False)
        
        print("特征扰动分析完成，结果已保存")
        
        return results_df
    
    def _visualize_perturbation_results(self, results_df):
        """可视化特征扰动结果"""
        # 选择前10个最重要的特征（基于平均RMSE变化）
        feature_sensitivity = results_df.groupby('feature')['rmse_change'].apply(lambda x: np.abs(x).mean()).sort_values(ascending=False)
        top_features = feature_sensitivity.head(10).index
        
        # 创建热力图
        pivot_data = results_df[results_df['feature'].isin(top_features)].pivot(
            index='feature', columns='perturbation', values='rmse_change'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlBu_r', center=0)
        plt.title('特征扰动敏感性热力图')
        plt.xlabel('扰动水平')
        plt.ylabel('特征')
        plt.tight_layout()
        plt.savefig('Q2/feature_perturbation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def hyperparameter_grid_search(self):
        """超参数邻域网格搜索"""
        print("执行超参数邻域网格搜索...")
        
        if hasattr(self.X, '__len__') and len(self.X) == 0:
            print("错误：数据未准备好")
            return None, None
        
        # 定义超参数搜索空间
        param_grid = {
            'n_estimators': [500, 750, 1000],
            'learning_rate': [0.005, 0.01, 0.02],
            'max_depth': [6, 8, 10],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.5, 0.6, 0.7]
        }
        
        # 执行网格搜索
        best_score = -np.inf
        best_params = None
        results = []
        
        from itertools import product
        
        # 生成所有参数组合
        param_combinations = list(product(*param_grid.values()))
        
        print(f"共 {len(param_combinations)} 种参数组合需要测试")
        
        for i, params in enumerate(param_combinations):
            param_dict = dict(zip(param_grid.keys(), params))
            
            # 创建模型
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                **param_dict
            )
            
            # 执行5折交叉验证
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, self.X, self.y, cv=kf, scoring='r2')
            
            mean_score = cv_scores.mean()
            std_score = cv_scores.std()
            
            results.append({
                **param_dict,
                'mean_r2': mean_score,
                'std_r2': std_score
            })
            
            # 更新最佳参数
            if mean_score > best_score:
                best_score = mean_score
                best_params = param_dict
            
            if (i + 1) % 10 == 0:
                print(f"已完成 {i + 1}/{len(param_combinations)} 种参数组合测试")
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 可视化结果
        self._visualize_grid_search_results(results_df)
        
        # 保存结果
        results_df.to_csv('Q2/hyperparameter_grid_search_results.csv', index=False)
        
        print("超参数网格搜索完成")
        print(f"最佳参数: {best_params}")
        print(f"最佳R²分数: {best_score:.4f}")
        
        return results_df, best_params
    
    def _visualize_grid_search_results(self, results_df):
        """可视化网格搜索结果"""
        # 选择最重要的几个参数进行可视化
        important_params = ['n_estimators', 'learning_rate', 'max_depth']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, param in enumerate(important_params):
            # 计算每个参数值的平均R2分数
            param_performance = results_df.groupby(param)['mean_r2'].agg(['mean', 'std']).reset_index()
            
            # 绘制柱状图
            axes[i].bar(param_performance[param], param_performance['mean'], 
                       yerr=param_performance['std'], alpha=0.7)
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('平均R²分数')
            axes[i].set_title(f'{param}对模型性能的影响')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Q2/hyperparameter_grid_search_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("生成综合鲁棒性分析报告...")
        
        if len(self.df) == 0 or (hasattr(self.X, '__len__') and len(self.X) == 0) or self.model is None:
            print("错误：数据或模型未准备好")
            return None
        
        # 创建报告
        report = {
            'data_info': {
                'sample_count': len(self.df),
                'feature_count': self.X.shape[1],
                'target_mean': float(self.y.mean()),
                'target_std': float(self.y.std())
            },
            'model_info': {
                'model_type': 'XGBoost Regressor',
                'base_params': self.model.get_params()
            }
        }
        
        # 保存报告
        import json
        with open('Q2/robustness_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print("综合分析报告已保存")
        
        return report
    
    def run_full_analysis(self):
        """运行完整的鲁棒性分析"""
        print("开始完整的模型鲁棒性分析...")
        
        # 1. 加载和预处理数据
        self.load_and_preprocess_data()
        
        # 2. 训练基础模型
        self.train_base_model()
        
        # 3. 交叉验证分析
        cv_results = self.cross_validation_analysis()
        
        # 4. SHAP值分析
        shap_values = self.shap_analysis()
        
        # 5. 特征消融研究
        ablation_results = self.feature_ablation_study()
        
        # 6. 特征扰动分析
        perturbation_results = self.feature_perturbation_analysis()
        
        # 7. 超参数网格搜索
        grid_search_results, best_params = self.hyperparameter_grid_search()
        
        # 8. 生成综合报告
        report = self.generate_comprehensive_report()
        
        print("完整的模型鲁棒性分析已完成！")
        
        return {
            'cv_results': cv_results,
            'shap_values': shap_values,
            'ablation_results': ablation_results,
            'perturbation_results': perturbation_results,
            'grid_search_results': grid_search_results,
            'best_params': best_params,
            'report': report
        }


def main():
    """主函数"""
    # 创建鲁棒性分析器
    analyzer = ModelRobustnessAnalyzer('../input_data/df_movies_train.csv')
    
    # 运行完整分析
    results = analyzer.run_full_analysis()
    
    print("\n=== 鲁棒性分析总结 ===")
    print(f"交叉验证平均RMSE: {results['cv_results']['RMSE'].mean():.4f}")
    print(f"交叉验证平均R²: {results['cv_results']['R2'].mean():.4f}")
    print(f"最敏感的特征（基于消融研究）: {results['ablation_results'].iloc[0]['feature']}")
    print(f"最佳超参数: {results['best_params']}")


if __name__ == "__main__":
    main()