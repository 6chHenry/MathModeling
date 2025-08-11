# 影院排片优化程序详细解释

## 1. 程序设计思想

### 1.1 核心思想
本程序采用**混合整数线性规划（MILP）**方法解决影院多放映厅排片优化问题。将复杂的排片决策转化为数学优化问题，通过定义决策变量、目标函数和约束条件，寻找最优的排片方案。

### 1.2 问题建模思路
```
实际问题 → 数学模型 → 优化求解 → 结果输出
    ↓           ↓         ↓         ↓
排片决策 → MILP模型 → COPT求解器 → 排片计划
```

### 1.3 建模核心要素
- **决策变量**：是否在特定时间、特定放映厅播放特定版本的电影
- **目标函数**：最大化影院净收益
- **约束条件**：运营规则和物理限制

## 2. 程序架构设计

### 2.1 类设计模式
采用**面向对象设计**，将排片优化器封装为`CinemaSchedulingOptimizer`类：

```python
class CinemaSchedulingOptimizer:
    # 数据初始化和参数设置
    def __init__(self, cinema_file, movies_file)
    
    # 辅助计算函数
    def _generate_time_slots(self)
    def _calculate_ticket_price(self)
    def _calculate_attendance(self)
    # ... 其他辅助函数
    
    # 核心优化函数
    def _optimize_with_copt(self)
```

### 2.2 模块化设计
- **数据处理模块**：读取和预处理输入数据
- **参数设置模块**：配置优化参数和约束条件
- **建模模块**：构建MILP数学模型
- **求解模块**：调用COPT求解器
- **结果处理模块**：提取和格式化输出结果

## 3. 详细代码解释

### 3.1 初始化部分

```python
def __init__(self, cinema_file, movies_file):
    # 数据读取
    self.cinema_df = pd.read_csv(cinema_file)      # 放映厅信息
    self.movies_df = pd.read_csv(movies_file)      # 电影信息
    
    # 时间系统设计
    self.start_hour = 10    # 营业开始：10:00
    self.end_hour = 27      # 营业结束：次日03:00（用27表示跨日）
    self.time_slots = self._generate_time_slots()  # 生成15分钟间隔时间点
```

**设计亮点**：
- 使用27小时制巧妙处理跨日营业时间
- 15分钟间隔时间段满足整刻钟开始要求

### 3.2 时间处理机制

```python
def _generate_time_slots(self):
    """生成时间段列表，每15分钟一个时间点"""
    slots = []
    current_hour = self.start_hour  # 从10点开始
    current_minute = 0
    
    while current_hour < self.end_hour:  # 到27点（次日3点）结束
        slots.append(f"{current_hour:02d}:{current_minute:02d}")
        current_minute += 15
        if current_minute >= 60:
            current_minute = 0
            current_hour += 1
    return slots
```

**时间转换机制**：
```python
def _convert_to_display_time(self, time_slot):
    """将内部27小时制转换为标准24小时制"""
    hour, minute = map(int, time_slot.split(':'))
    if hour >= 24:
        display_hour = hour - 24  # 25:30 → 01:30
        return f"{display_hour:02d}:{minute:02d}"
    else:
        return time_slot
```

### 3.3 核心计算函数

#### 票价计算
```python
def _calculate_ticket_price(self, movie_id, version, is_prime_time=False):
    movie = self.movies_df[self.movies_df['id'] == movie_id].iloc[0]
    basic_price = movie['basic_price']
    
    # 版本价格系数
    if version == '2D':      price = basic_price
    elif version == '3D':    price = basic_price * 1.2
    elif version == 'IMAX':  price = basic_price * 1.23
    
    # 黄金时段加价
    if is_prime_time:
        price *= self.prime_time_multiplier  # × 1.3
    
    return price
```

#### 观影人数计算
```python
def _calculate_attendance(self, capacity, rating):
    """根据评分计算实际观影人数"""
    return math.floor(capacity * rating / 10)
```
**公式来源**：题目给定的上座率计算公式

#### 成本计算
```python
def _calculate_cost(self, capacity, version):
    """计算单场播放成本"""
    version_coeff = self.version_coeff[version]  # 版本系数
    # cost = version_coeff × capacity × basic_cost + fixed_cost
    return version_coeff * capacity * self.basic_cost + self.fixed_cost
```

### 3.4 MILP模型构建

#### 决策变量定义
```python
# 四维决策变量：x[放映厅][电影ID][版本][时间段]
x = {}
for room in rooms:
    for movie_id in movies:
        for version in versions:
            for time_slot in time_slots:
                # 检查可行性（设备支持、时间约束等）
                if self._is_feasible(room, movie_id, version, time_slot):
                    x[room][movie_id][version][time_slot] = Binary_Variable
```

**变量含义**：
- `x[R01][12345][2D][10:00] = 1` 表示在R01放映厅10:00播放电影12345的2D版本
- `x[R01][12345][2D][10:00] = 0` 表示不进行此安排

#### 目标函数构建
```python
obj_expr = 0
for each_decision_variable:
    # 计算该场次的净收益
    ticket_revenue = ticket_price × attendance
    net_revenue = ticket_revenue × (1 - sharing_rate)
    cost = calculate_cost()
    net_profit = net_revenue - cost
    
    # 累加到目标函数
    obj_expr += net_profit × decision_variable

model.setObjective(obj_expr, MAXIMIZE)  # 最大化净收益
```

### 3.5 约束条件实现

#### 约束1：时间冲突约束
```python
for room in rooms:
    for time_slot in time_slots:
        overlapping_vars = []  # 收集可能冲突的变量
        
        for movie_show in all_possible_shows:
            # 检查该场次是否与当前时间段冲突
            if time_conflict_exists(movie_show, time_slot):
                overlapping_vars.append(decision_variable)
        
        # 同一时间段最多只能选择一个
        if overlapping_vars:
            model.addConstr(sum(overlapping_vars) <= 1)
```

**冲突判定逻辑**：
```python
# 检查时间重叠（包含15分钟清理时间）
if start_minutes <= current_minutes < end_minutes + 15:
    # 存在冲突
```

#### 约束2：版本播放时长限制
```python
for version in ['3D', 'IMAX']:
    total_duration_vars = []
    
    for each_show_of_this_version:
        runtime = round_up_to_30(movie_runtime)
        total_duration_vars.append(runtime × decision_variable)
    
    # 总播放时长约束
    model.addConstr(sum(total_duration_vars) <= max_limit)
    model.addConstr(sum(total_duration_vars) >= min_limit)
```

#### 约束3：题材播放次数限制
```python
for genre in ['Animation', 'Horror', 'Action', 'Drama']:
    genre_vars = []
    
    for each_show:
        if movie_belongs_to_genre(movie, genre):
            genre_vars.append(decision_variable)
    
    # 播放次数约束
    if has_max_limit:
        model.addConstr(sum(genre_vars) <= max_limit)
    if has_min_limit:
        model.addConstr(sum(genre_vars) >= min_limit)
```

#### 约束4：设备连续运行时长限制
```python
for room in rooms:
    # 每15分钟滑动一次的9小时窗口
    for window_start in sliding_windows:
        window_duration_vars = []
        
        for each_show_in_room:
            # 计算播放时间与窗口的重叠时长
            overlap_duration = calculate_overlap(show_time, window_time)
            if overlap_duration > 0:
                window_duration_vars.append(overlap_duration × decision_variable)
        
        # 9小时内累计播放不超过7小时（420分钟）
        model.addConstr(sum(window_duration_vars) <= 420)
```

### 3.6 求解过程

```python
def _optimize_with_copt(self):
    # 1. 创建COPT环境
    env = cp.Envr()
    model = env.createModel("Cinema_Scheduling")
    
    # 2. 创建决策变量
    create_decision_variables()
    
    # 3. 设置目标函数
    set_objective_function()
    
    # 4. 添加约束条件
    add_all_constraints()
    
    # 5. 设置求解参数
    model.setParam(COPT.Param.TimeLimit, 300)    # 5分钟限制
    model.setParam(COPT.Param.RelGap, 0.01)      # 1%相对差距
    
    # 6. 求解
    model.solve()
    
    # 7. 提取结果
    if model.status == OPTIMAL:
        extract_and_format_results()
    
    return results
```

### 3.7 结果处理

```python
# 提取最优解
schedule_results = []
for each_decision_variable:
    if variable.value > 0.5:  # 二进制变量接近1
        # 计算该场次的详细信息
        attendance = calculate_attendance()
        display_time = convert_time_format()
        
        schedule_results.append({
            'room': room_id,
            'showtime': display_time,
            'id': movie_id,
            'version': version,
            'attendance': attendance
        })

# 按房间和时间排序
schedule_results.sort(key=lambda x: (x['room'], time_sort_key(x['showtime'])))
```

## 4. 算法优势与特点

### 4.1 数学建模优势
- **全局最优**：MILP保证在可行域内找到全局最优解
- **约束处理**：能够精确处理复杂的运营约束
- **可扩展性**：易于添加新的约束条件

### 4.2 程序设计特点
- **模块化设计**：各功能模块清晰分离，易于维护
- **数据驱动**：通过CSV文件输入，支持不同数据集
- **时间处理**：巧妙的27小时制处理跨日营业
- **约束验证**：完整的可行性检查机制

### 4.3 性能优化策略
- **预处理过滤**：在创建变量前过滤不可行的组合
- **稀疏矩阵**：只创建必要的决策变量
- **求解参数**：设置合理的时间限制和精度要求

## 5. 程序流程总结

```
输入数据 → 参数初始化 → 时间段生成 → 可行性检查 → 变量创建 
    ↓
目标函数构建 → 约束添加 → 求解器调用 → 结果提取 → 格式转换
    ↓
排片计划输出 → 约束验证 → 统计分析 → CSV导出
```

这个程序通过严谨的数学建模和高效的算法实现，将复杂的影院排片问题转化为可计算的优化问题，为影院提供科学的排片决策支持。