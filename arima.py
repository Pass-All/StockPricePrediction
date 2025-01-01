import os
import glob
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="statsmodels")

plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_directories(stock_code):
    """创建必要的文件夹"""
    base_dir = "ARIMA"
    stock_dir = os.path.join(base_dir, stock_code)
    
    # 创建基础目录
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # 创建股票特定目录
    if not os.path.exists(stock_dir):
        os.makedirs(stock_dir)
        
    return base_dir, stock_dir

def clean_stock_data(df):
    missing_values = df.isnull().sum()
    invalid_prices = ~((df['收盘价(元)'] >= df['最低价(元)']) & 
                      (df['收盘价(元)'] <= df['最高价(元)']))
    
    if invalid_prices.any():
        df.loc[invalid_prices, '收盘价(元)'] = df['收盘价(元)'].shift(1)
        df.loc[invalid_prices, '收盘价(元)'].fillna(df['收盘价(元)'].shift(-1))
    
    return df

def check_stationarity_all(data):
    """综合检查序列的平稳性"""
    adf_result = adfuller(data)
    dftest_result = adfuller(data, regression='ct')
    kpss_result = kpss(data, regression='c')
    lb_result = acorr_ljungbox(data, lags=[10], return_df=True)
    
    is_stationary = (adf_result[1] < 0.05 and 
                    dftest_result[1] < 0.05 and 
                    kpss_result[1] > 0.05 and 
                    lb_result["lb_pvalue"].values[0] < 0.05)
                    
    return is_stationary

def grid_search_parameters(train_data, d):
    """网格搜索ARIMA最佳参数"""
    p_range = range(0, 5)
    q_range = range(0, 5)
    results_list = []
    
    total_combinations = len(p_range) * len(q_range)
    current = 0
    
    print("\n开始网格搜索...")
    print("-" * 50)
    print(f"{'p':>3} {'q':>3} {'AIC':>12}")
    print("-" * 50)
    
    for p in p_range:
        for q in q_range:
            current += 1
            try:
                model = ARIMA(train_data, order=(p, d, q))
                results_fit = model.fit()
                aic = results_fit.aic
                print(f"{p:>3} {q:>3} {aic:>12.4f}")
                
                results_list.append({
                    'p': p,
                    'd': d,
                    'q': q,
                    'aic': aic
                })
            except:
                print(f"{p:>3} {q:>3} {'失败':>12}")
                continue
    
    results = pd.DataFrame(results_list)
    results = results.sort_values('aic')
    best_params = results.iloc[0]
    
    if best_params['p'] == 0 and best_params['q'] == 0:
        best_params = results.iloc[1]
    
    print("\n" + "=" * 50)
    print(f"最优参数: p={int(best_params['p'])}, d={d}, q={int(best_params['q'])}")
    print(f"最优AIC值: {best_params['aic']:.4f}")
    print("=" * 50 + "\n")
        
    return (int(best_params['p']), int(best_params['d']), int(best_params['q']))

def save_plot(fig, stock_code, round_num, plot_type, base_dir="ARIMA"):
    """保存图表到指定目录"""
    # 创建保存路径
    save_dir = os.path.join(base_dir, stock_code)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 根据类型确定文件名
    filename = f"{stock_code}_{plot_type}"
    if round_num is not None:
        filename += f"_round{round_num}"
    filename += ".png"
    
    # 保存图表
    fig.savefig(os.path.join(save_dir, filename))
    plt.close(fig)

def plot_and_save_timeseries(dates, data, title, stock_code, round_num=None, plot_type=""):
    """绘制并保存时序图"""
    fig = plt.figure(figsize=(12, 6))
    plt.plot(dates, data)
    plt.title(title)
    plt.xlabel('日期')
    plt.ylabel('标准化后的收盘价')
    
    # 修改x轴刻度显示
    n_ticks = 10  # 想要显示的刻度数
    step = len(dates) // n_ticks
    plt.xticks(range(0, len(dates), step), dates[::step], rotation=45)
    
    plt.grid(True)
    plt.tight_layout()
    save_plot(fig, stock_code, round_num, plot_type)
    plt.close()

def process_single_stock(file_path, rounds=3):
    """处理单个股票数据"""
    stock_code = os.path.basename(file_path).split('.')[0]

    print(f"\n正在处理股票: {stock_code}")
    print("1. 数据预处理...")

    # 读取和预处理数据
    df = pd.read_excel(file_path, skiprows=2, nrows=1690)
    df_cleaned = clean_stock_data(df)
    df_cleaned.index = pd.to_datetime(df_cleaned['交易日期'])
    prices = np.array(df_cleaned['收盘价(元)'])
    dates = np.array(df_cleaned['交易日期'])
    prices_log = np.log(prices)
    scaler = StandardScaler()
    prices_log_scaled = scaler.fit_transform(prices_log.reshape(-1, 1)).ravel()
    
    # 9:1划分数据
    n = len(prices_log_scaled)
    train_size = int(n * 0.9)
    train_data = prices_log_scaled[:train_size]
    test_data = prices_log_scaled[train_size:]
    train_dates = dates[:train_size]
    test_dates = dates[train_size:]
    
    # 绘制并保存原始时序图
    plot_and_save_timeseries(dates, prices_log_scaled, 
                             f'{stock_code} 原始时序图', 
                             stock_code, plot_type="original")

    # 检查平稳性和差分
    print("进行平稳性检验...")
    d = 0
    curr_data = train_data.copy()
    while d < 2:
        if check_stationarity_all(curr_data):
            break
        print(f"执行{d+1}阶差分...")
        d += 1
        curr_data = np.diff(curr_data)
    
    # 网格搜索最优参数
    print("网格搜索最优参数...")
    best_order = grid_search_parameters(train_data, d)
    print(f"最优参数: p={best_order[0]}, d={best_order[1]}, q={best_order[2]}")
    
    # 训练模型并预测
    print("训练模型并预测...")
    model = ARIMA(train_data, order=best_order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test_data))
    
    # 计算性能指标
    rae = np.sum(np.abs(test_data - forecast)) / np.sum(np.abs(test_data - np.mean(test_data)))
    rse = np.sum((test_data - forecast)**2) / np.sum((test_data - np.mean(test_data))**2)
    
    # 绘制预测结果
    plt.figure(figsize=(15, 8))
    plt.plot(test_dates, test_data, label='测试数据')
    plt.plot(test_dates, forecast, label='预测数据', linestyle='--')
    plt.title(f'{stock_code} 预测结果对比')
    plt.xlabel('日期')
    plt.ylabel('标准化收盘价')
    
    n_ticks = 10
    step = len(test_dates) // n_ticks
    plt.xticks(range(0, len(test_dates), step), test_dates[::step], rotation=45)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    save_plot(plt.gcf(), stock_code, None, plot_type="forecast")
    plt.close()
    
    return {
        'stock_code': stock_code,
        'rae': rae,
        'rse': rse
    }

# def train_arima_with_validation(train_data, val_data, d, p_range=(0,5), q_range=(0,5)):
    """使用验证集训练ARIMA模型
    Args:
        train_data: 训练数据
        val_data: 验证数据 
        d: 已确定的差分阶数
        p_range: AR阶数搜索范围
        q_range: MA阶数搜索范围
    """
    best_val_metrics = {'rae': float('inf'), 'rse': float('inf')}
    best_order = None
    best_model = None
    
    # 只搜索p,q参数
    for p in range(*p_range):
        for q in range(*q_range):
            try:
                # 训练模型
                model = ARIMA(train_data, order=(p, d, q))
                results = model.fit()
                
                # 验证集预测
                # history = list(train_data)
                # predictions = []
                # for t in range(len(val_data)):
                #     model_fit = ARIMA(history, order=(p, d, q)).fit()
                #     yhat = model_fit.forecast(steps=1)[0]
                #     predictions.append(yhat)
                #     history.append(val_data[t])
                
                # 一次性预测整个验证集
                predictions = results.forecast(steps=len(val_data))
                    
                # 计算验证集指标
                val_rae = np.sum(np.abs(val_data - predictions)) / np.sum(np.abs(val_data - np.mean(val_data)))
                val_rse = np.sum((val_data - predictions)**2) / np.sum((val_data - np.mean(val_data))**2)
                
                # 更新最佳模型
                if val_rae < best_val_metrics['rae']:
                    best_val_metrics = {'rae': val_rae, 'rse': val_rse}
                    best_order = (p, d, q)
                    best_model = results
                    
                print(f"ARIMA({p},{d},{q}) - VAL RAE: {val_rae:.4f}, RSE: {val_rse:.4f}")
                    
            except:
                continue
                
    return best_model, best_order, best_val_metrics

def main():
    # 指定股票数据所在的文件夹路径
    folder_path = "每日行情/"
    excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    
    if not excel_files:
        print("未找到任何Excel文件，请检查文件夹路径。")
        return
    
    print(f"\n总共发现{len(excel_files)}个股票文件")
    # 用于汇总所有股票的评估结果
    all_results = []
    
    # 处理每个股票文件
    for file in tqdm(excel_files, desc="总体进度"):
        file_name = os.path.basename(file)
        print(f"\n{'='*50}")
        print(f"开始处理: {file_name}")
        
        try:
            # 处理单个股票
            result = process_single_stock(file)
            all_results.append({
                '股票文件': file_name,
                'RAE': result['rae'],
                'RSE': result['rse']
            })
            
            print(f"\n处理完成: {file_name}")
            print(f"RAE={result['rae']:.4f}, RSE={result['rse']:.4f}")
            print(f"{'='*50}")
            
        except Exception as e:
            print(f"\n处理{file_name}时出错: {str(e)}")
            print(f"{'='*50}")
            continue
    
    print("\n所有股票处理完成!")
    # 保存所有结果
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("所有股票ARIMA结果.csv", index=False)
    print("已保存结果到: 所有股票ARIMA结果.csv")

if __name__ == "__main__":
    main()
