# Q3 参数扰动测试组件文档

## 1. 组件概述

### 1.1 组件名称
ParameterPerturbationAnalyzer（参数扰动分析器）

### 1.2 主要功能
该组件提供全面的参数扰动测试功能，用于评估电影院排片优化系统在不同参数设置和约束条件下的表现稳定性。主要功能包括：
- 参数扰动测试（±变化）
- 约束边界测试（±20%场景）
- 随机返回噪声重采样30次评估解的稳定性
- Gap指标和参数变化率分析

### 1.3 应用场景
该组件适用于电影院排片优化系统的鲁棒性评估，帮助理解排片方案对参数变化、约束调整的敏感程度，以及系统在不确定条件下的表现稳定性。特别适用于：
- 排片规则参数优化
- 约束条件合理性评估
- 排片方案稳定性分析
- 运营策略调整影响评估

## 2. 技术架构

### 2.1 依赖库
- pandas: 数据处理和分析
- numpy: 数值计算
- matplotlib & seaborn: 数据可视化
- datetime: 时间处理
- collections: 数据结构（defaultdict）
- json: 规则文件加载

### 2.2 核心类结构
```python
class ParameterPerturbationAnalyzer:
    def __init__(self, schedule_file, movies_file, rules_file=None)
    def load_data(self)
    def parameter_perturbation_test(self)
    def constraint_boundary_test(self)
    def noise_resampling_test(self)
    def generate_comprehensive_report(self)
    def run_full_analysis(self)
```

### 2.3 数据流程
1. 排片数据和电影数据加载
2. 规则配置和预处理
3. 多维度参数扰动测试
4. 约束边界测试
5. 噪声重采样稳定性评估
6. 结果可视化与报告生成

## 3. 功能模块详解

### 3.1 数据加载与预处理模块

#### 3.1.1 数据输入
- **排片数据**：包含场次信息（电影ID、版本、放映时间、放映厅等）
- **电影数据**：包含电影信息（ID、时长、题材、语言等）
- **规则数据**：可选JSON格式规则配置文件

#### 3.1.2 数据预处理
- 电影时长向上取整到30分钟倍数
- 题材列表处理和标准化
- 时间格式解析和转换

### 3.2 参数扰动测试模块

#### 3.2.1 测试参数
- **min_gap**：最小间隔时间（分钟）
- **version_caps**：版本时长限制（3D、IMAX）
- **genre_caps**：题材播放次数限制

#### 3.2.2 扰动水平
- 0.8倍（减少20%）
- 0.9倍（减少10%）
- 1.1倍（增加10%）
- 1.2倍（增加20%）

#### 3.2.3 测试方法
1. 记录基础排片性能指标
2. 逐一修改参数值
3. 重新计算性能指标
4. 计算变化量和变化率
5. 分析参数敏感性

#### 3.2.4 评估指标
- 版本时长变化
- 题材次数变化
- 总场次变化
- 黄金时段场次变化
- 平均间隔时间变化

#### 3.2.5 输出结果
- 参数扰动敏感性曲线图
- 扰动结果数据表
- 结果保存为CSV和PNG格式

### 3.3 约束边界测试模块

#### 3.3.1 测试约束
- **version_total_caps**：版本总时长限制
- **genre_caps**：题材播放次数限制

#### 3.3.2 边界水平
- 0.8倍（减少20%）
- 1.2倍（增加20%）

#### 3.3.3 测试方法
1. 记录基础排片性能指标
2. 调整约束边界值
3. 重新计算性能指标
4. 计算约束违反情况
5. 评估边界变化影响

#### 3.3.4 约束违反检测
- 版本时长约束违反检测
- 题材次数约束违反检测
- 违反次数统计和分析

#### 3.3.5 输出结果
- 约束违反情况柱状图
- 边界百分比对性能影响图
- 结果保存为CSV和PNG格式

### 3.4 噪声重采样测试模块

#### 3.4.1 噪声类型
- **时间噪声**：放映时间±15分钟随机扰动
- **版本噪声**：随机更换电影版本（2D/3D/IMAX）
- **放映厅噪声**：随机更换放映厅

#### 3.4.2 测试方法
1. 记录基础排片性能指标
2. 添加随机噪声生成新排片方案
3. 计算噪声排片性能指标
4. 计算Gap指标（绝对差异）
5. 重复30次评估稳定性

#### 3.4.3 Gap指标计算
- 总场次Gap
- 黄金时段场次Gap
- 平均间隔时间Gap
- 版本时长Gap
- 题材次数Gap
- 总Gap（各项Gap之和）

#### 3.4.4 稳定性评估
- 计算各Gap指标的统计量（均值、标准差、最小值、最大值）
- 计算变异系数（CV）评估稳定性
- 稳定性等级划分：
  - 高稳定性：CV < 0.1
  - 中等稳定性：0.1 ≤ CV < 0.3
  - 低稳定性：CV ≥ 0.3

#### 3.4.5 输出结果
- Gap指标分布直方图
- 各指标Gap箱线图
- 变化率分布图
- 结果保存为CSV和PNG格式

### 3.5 综合分析模块

#### 3.5.1 关键发现分析
- 最敏感参数识别
- 最常违反约束识别
- 平均总Gap计算
- 解稳定性评估

#### 3.5.2 敏感性分析
- 计算每个参数的平均绝对变化率
- 识别对系统影响最大的参数
- 提供参数优化建议

#### 3.5.3 稳定性分析
- 基于噪声重采样结果评估解的稳定性
- 计算总Gap的变异系数
- 提供稳定性等级评定

#### 3.5.4 输出结果
- 综合分析报告（JSON格式）
- 包含所有测试结果的摘要
- 关键发现和建议

## 4. 使用方法

### 4.1 基本使用
```python
# 创建分析器实例
analyzer = ParameterPerturbationAnalyzer(
    schedule_file='Q3/df_result_2_copt_ours_new.csv',
    movies_file='Q3/df_movies_schedule_ours_new.csv'
)

# 运行完整分析
results = analyzer.run_full_analysis()

# 查看结果
print(f"参数扰动测试: {len(results['perturbation_results'])} 个测试场景")
print(f"约束边界测试: {len(results['constraint_results'])} 个测试场景")
print(f"噪声重采样测试: {len(results['noise_results'])} 次重采样")
print(f"平均总Gap: {results['noise_results']['total_gap'].mean():.2f}")
```

### 4.2 单独使用各模块
```python
# 仅进行参数扰动测试
perturbation_results = analyzer.parameter_perturbation_test()

# 仅进行约束边界测试
constraint_results = analyzer.constraint_boundary_test()

# 仅进行噪声重采样测试
noise_results, noise_stats = analyzer.noise_resampling_test()
```

### 4.3 自定义规则配置
```python
# 创建自定义规则文件
rules = {
    "open_time": "10:00",
    "close_time": "03:00",
    "min_gap": 15,
    "golden_start": "18:00",
    "golden_end": "21:00",
    "version_coeff": {"2D": 1.0, "3D": 1.1, "IMAX": 1.15},
    "version_total_caps": {"3D": 1200, "IMAX": 1500},
    "genre_caps": {"Animation": (1, 5), "Horror": (0, 3), "Action": (2, 6), "Drama": (1, 6)},
    "genre_time_limits": {
        "Animation": (None, "19:00"),
        "Family": (None, "19:00"),
        "Horror": ("21:00", None),
        "Thriller": ("21:00", None),
    },
}

# 保存为JSON文件
import json
with open('custom_rules.json', 'w', encoding='utf-8') as f:
    json.dump(rules, f, ensure_ascii=False, indent=2)

# 使用自定义规则
analyzer = ParameterPerturbationAnalyzer(
    schedule_file='schedule.csv',
    movies_file='movies.csv',
    rules_file='custom_rules.json'
)
```

## 5. 输出文件说明

### 5.1 结果文件
- `parameter_perturbation_analysis.png`: 参数扰动结果可视化
- `parameter_perturbation_results.csv`: 参数扰动数据
- `constraint_boundary_analysis.png`: 约束边界结果可视化
- `constraint_boundary_results.csv`: 约束边界数据
- `noise_resampling_analysis.png`: 噪声重采样结果可视化
- `noise_resampling_results.csv`: 噪声重采样数据
- `parameter_perturbation_report.json`: 综合分析报告

### 5.2 文件格式说明
- PNG文件：可视化图表，300 DPI高清保存
- CSV文件：结构化数据，便于进一步分析
- JSON文件：综合报告，包含所有关键发现

## 6. 参数配置

### 6.1 默认规则参数
- **营业时间**：10:00-03:00（次日）
- **最小间隔**：15分钟
- **黄金时段**：18:00-21:00
- **版本系数**：2D(1.0), 3D(1.1), IMAX(1.15)
- **版本时长限制**：3D(1200分钟), IMAX(1500分钟)
- **题材限制**：
  - Animation: 1-5次
  - Horror: 0-3次
  - Action: 2-6次
  - Drama: 1-6次
- **题材时间限制**：
  - Animation/Family: 不晚于19:00
  - Horror/Thriller: 不早于21:00

### 6.2 测试参数
- 扰动水平：[0.8, 0.9, 1.1, 1.2]
- 边界百分比：[0.8, 1.2]
- 噪声重采样次数：30次
- 噪声水平：0.1（10%）

## 7. 性能考虑

### 7.1 计算复杂度
- 参数扰动测试：O(n_parameters × n_perturbation_levels × n_shows)
- 约束边界测试：O(n_constraints × n_boundary_levels × n_shows)
- 噪声重采样测试：O(n_samples × n_shows)

### 7.2 优化建议
- 对于大规模排片数据，可考虑减少扰动水平数量
- 可并行化处理不同参数的扰动测试
- 结果缓存机制可避免重复计算

## 8. 扩展性

### 8.1 自定义扩展
- 可添加新的参数类型进行扰动测试
- 可扩展约束类型和边界测试
- 可集成其他稳定性评估方法

### 8.2 集成建议
- 可作为排片优化系统的标准评估组件
- 可与参数调优流程集成
- 支持自动化测试和报告生成

## 9. 注意事项

### 9.1 使用限制
- 要求数据格式与预期一致
- 需要合理的排片数据量以确保测试的可靠性
- 扰动水平应根据实际业务场景调整

### 9.2 数据要求
- 输入数据应为CSV格式
- 排片数据必须包含：id、version、showtime、room等字段
- 电影数据必须包含：id、runtime、genres等字段
- 建议排片数据量大于100条记录

### 9.3 依赖环境
- Python 3.6+
- 所需依赖库版本见requirements.txt
- 建议使用虚拟环境管理依赖

## 10. 故障排除

### 10.1 常见问题
1. **数据加载失败**：检查文件路径和格式
2. **规则加载失败**：检查JSON格式和字段名称
3. **时间解析错误**：确保时间格式为HH:MM
4. **可视化错误**：确保matplotlib后端配置正确

### 10.2 调试建议
- 使用print语句跟踪执行流程
- 检查中间数据的维度和类型
- 验证规则参数的合理性
- 确保所有必需字段都存在

### 10.3 性能问题
- 减少扰动水平和边界测试的数量
- 优化噪声重采样次数
- 考虑数据采样以提高处理速度

## 11. 结果解释指南

### 11.1 参数扰动结果解释
- **变化率**：正值表示增加，负值表示减少
- **敏感性**：变化率越大，参数越敏感
- **趋势**：观察参数变化对系统影响的趋势

### 11.2 约束边界结果解释
- **违反次数**：表示约束的严格程度
- **边界影响**：边界变化对系统性能的影响程度
- **约束合理性**：基于违反次数评估约束设置的合理性

### 11.3 噪声重采样结果解释
- **Gap值**：绝对差异，越小表示越稳定
- **变异系数**：小于0.1表示高稳定性
- **分布形状**：正态分布表示系统稳定性好

### 11.4 综合报告解读
- **最敏感参数**：需要重点监控和优化的参数
- **最常违反约束**：可能需要调整的约束条件
- **解稳定性**：整体系统稳定性的评估
- **建议**：基于测试结果的具体改进建议