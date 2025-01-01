# 导入必要的库
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import copy

# 定义LSTM模型类
class LSTMModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_layers=3, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        if len(out.shape) == 3:
            out = out[:, -1, :]
        else:
            raise ValueError(f"Unexpected tensor shape: {out.shape}")
        out = self.fc(out)
        return out

# 定义训练LSTM模型的函数
def train_lstm(train_loader, val_loader, model, criterion, optimizer, 
               num_epochs, patience=10, threshold=1e-4):
    """
    训练LSTM模型，包含早停和学习率调度
    Args:
        patience: 早停等待轮数
        threshold: 最小改善阈值
    """
    # 初始化学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=10,
        threshold=1e-4  # 将 min_delta 改为 threshold
    )
    
    # 初始化早停所需变量
    best_val_loss = float('inf')
    best_model = None
    counter = 0
    
    model.train()
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.float(), batch_y.float()
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.float(), batch_y.float()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss - threshold:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            
        # 每5轮打印一次损失
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
        # 如果验证损失连续patience轮未改善，则停止训练
        if counter >= patience:
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
    
    # 恢复最佳模型
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'stopped_epoch': len(train_losses)
    }

# 定义评估LSTM模型的函数
def evaluate_lstm(model, data_loader, scaler):
    model.eval()  # 设置模型为评估模式
    preds, trues = [], []  # 初始化预测值和真实值列表
    with torch.no_grad():  # 不计算梯度
        for batch_x, batch_y in data_loader:  # 遍历数据
            batch_x = batch_x.float()  # 转换数据类型为float
            outputs = model(batch_x)  # 获取模型输出
            preds.extend(outputs.numpy().flatten())  # 将输出添加到预测值列表
            trues.extend(batch_y.numpy().flatten())  # 将真实值添加到真实值列表

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()  # 逆变换预测值
    trues = scaler.inverse_transform(np.array(trues).reshape(-1, 1)).flatten()  # 逆变换真实值
    return preds, trues  # 返回预测值和真实值

# 定义计算评估指标的函数
def calculate_metrics(y_true, y_pred):
    rae = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))  # 计算相对绝对误差
    rse = np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)  # 计算相对平方误差
    return rae, rse  # 返回评估指标

# 定义添加技术指标的函数
def add_technical_features(df, window=7):
    """
    为DataFrame添加技术指标，如STD、EWMA、MOM、RSI、WR。
    window: 滚动窗口大小，默认为7
    """
    # 1. 收盘价的滚动标准差（STD）
    df['STD'] = df['收盘价(元)'].rolling(window).std()

    # 2. 指数加权移动平均值（EWMA）
    df['EWMA'] = df['收盘价(元)'].ewm(span=window).mean()

    # 3. 动量指标（MOM）: 当前收盘价 - 前window天收盘价
    df['MOM'] = df['收盘价(元)'] - df['收盘价(元)'].shift(window)

    # 4. 相对强弱指标（RSI）的简化版本
    # 计算涨跌幅
    change = df['收盘价(元)'].diff()
    gain = change.clip(lower=0).rolling(window).mean()
    loss = (-change.clip(upper=0)).rolling(window).mean()
    df['RSI'] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # 5. 威廉指标（WR）
    highest_high = df['最高价(元)'].rolling(window).max()
    lowest_low = df['最低价(元)'].rolling(window).min()
    df['WR'] = -100 * ((highest_high - df['收盘价(元)']) / (highest_high - lowest_low + 1e-9))

    # 将特征中的空值去除或填充
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    return df

def create_dataloader(features, targets, batch_size=32):
    x_tensor = torch.tensor(features, dtype=torch.float32) # 将特征转换为张量
    y_tensor = torch.tensor(targets, dtype=torch.float32) # 将目标转换为张量
    dataset = TensorDataset(x_tensor, y_tensor) # 创建数据集
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) # 创建数据加载器
    return loader

# 数据清洗
def clean_stock_data(df):
    # 移除逗号并将列转换为浮点数
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
        
    # 检查收盘价是否在最高价和最低价之间
    invalid_prices = ~((df['收盘价(元)'] >= df['最低价(元)']) & 
                      (df['收盘价(元)'] <= df['最高价(元)']))
    
    # 对无效价格进行修正(使用前一天或后一天的收盘价)
    if invalid_prices.any():
        df.loc[invalid_prices, '收盘价(元)'] = df['收盘价(元)'].shift(1)
        df.loc[invalid_prices, '收盘价(元)'].fillna(df['收盘价(元)'].shift(-1))
    
    return df

# 数据集划分
def prepare_data_forward_link(df, feature_cols, target_col):
    """
    前向链接法切分:
      1) 先整体 7:2:1 => a, b, c
      2) 再将 a => d, e, f (7:2:1)
      3) 得到 3 轮 (train, val, test)
    返回:
        {
          'forward_folds': [  # 三个轮次
              { 'train_X':..., 'train_y':..., 'val_X':..., 'val_y':..., 'test_X':..., 'test_y':... },
              ...
          ],
          'scaler': ...
        }
    """
    df = df.copy()
    df.sort_index(inplace=True)

    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y)

    n = len(X_scaled)
    # 划分 a, b, c
    a_end = int(n * 0.7)
    b_end = a_end + int(n * 0.2)
    # c段到末尾

    # a段再细分 d, e, f (7:2:1 -> 49%, 14%, 7%)
    a_len = a_end  # a段总长度
    d_end = int(a_len * 0.7)      # a的70%
    e_end = d_end + int(a_len*0.2)  # a的20%
    # f到 a_end

    # 具体数据切片
    d_X, d_y = X_scaled[:d_end], y_scaled[:d_end]
    e_X, e_y = X_scaled[d_end:e_end], y_scaled[d_end:e_end]
    f_X, f_y = X_scaled[e_end:a_end], y_scaled[e_end:a_end]
    b_X, b_y = X_scaled[a_end:b_end], y_scaled[a_end:b_end]
    c_X, c_y = X_scaled[b_end:], y_scaled[b_end:]

    # 生成三轮 (train, val, test)
    forward_folds = [
        {
            'train_X': d_X,
            'train_y': d_y,
            'val_X': e_X,
            'val_y': e_y,
            'test_X': f_X,
            'test_y': f_y
        },
        {
            'train_X': np.concatenate([d_X, e_X], axis=0),
            'train_y': np.concatenate([d_y, e_y], axis=0),
            'val_X': f_X,
            'val_y': f_y,
            'test_X': b_X,
            'test_y': b_y
        },
        {
            'train_X': np.concatenate([d_X, e_X, f_X], axis=0),
            'train_y': np.concatenate([d_y, e_y, f_y], axis=0),
            'val_X': b_X,
            'val_y': b_y,
            'test_X': c_X,
            'test_y': c_y
        }
    ]

    return {
        'forward_folds': forward_folds,
        'scaler': scaler
    }