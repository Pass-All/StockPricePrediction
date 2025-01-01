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
import multiprocessing as mp
from joblib import Parallel, delayed
from functools import partial
import gc
import matplotlib
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
    """优化后的网格搜索ARIMA最佳参数"""
    p_range = range(0, 5)
    q_range = range(0, 5)
    results_list = []
    
    # 并行处理网格搜索
    def fit_model(p, q, train_data, d):
        try:
            model = ARIMA(train_data, order=(p, d, q))
            results_fit = model.fit()
            return {'p': p, 'd': d, 'q': q, 'aic': results_fit.aic}
        except:
            return None
    
    # 使用joblib并行处理
    results = Parallel(n_jobs=-1)(
        delayed(fit_model)(p, q, train_data, d) 
        for p in p_range for q in q_range
    )
    
    # 过滤None结果并找到最佳参数
    results = [r for r in results if r is not None]
    if not results:
        return (1, d, 1)  # 默认参数
    
    best_params = min(results, key=lambda x: x['aic'])
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
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(fig, stock_code, round_num, plot_type)
    plt.close()

def plot_prediction_results(test_dates, test_data, forecast, stock_code, round_num):
    """绘制预测结果对比图"""
    # 确保数据长度匹配
    min_len = min(len(test_dates), len(test_data))
    test_dates = test_dates[:min_len]
    test_data = test_data[:min_len]
    forecast = forecast[:min_len]
    
    plt.figure(figsize=(15, 8))
    plt.plot(test_dates, test_data, label='测试数据')
    plt.plot(test_dates, forecast, label='预测数据', linestyle='--')
    plt.title(f'第{round_num + 1}轮预测结果对比')
    plt.xlabel('日期')
    plt.ylabel('标准化收盘价')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    save_plot(plt.gcf(), stock_code, round_num + 1, plot_type="forecast")
    plt.close()

def process_single_stock_optimized(file_path, rounds=3):
    """优化后的单个股票处理函数"""
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
    
    # 按7:2:1划分数据
    n = len(prices_log_scaled)
    a_end = int(n * 0.7)
    b_end = a_end + int(n * 0.2)
    
    # a段再细分为d,e,f
    a_len = a_end
    d_end = int(a_len * 0.7)
    e_end = d_end + int(a_len * 0.2)
    
    # 三轮训练数据
    rounds_data = [
        {
            'train': prices_log_scaled[:d_end],
            'test': prices_log_scaled[d_end:e_end]
        },
        {
            'train': prices_log_scaled[:e_end],
            'test': prices_log_scaled[e_end:a_end]
        },
        {
            'train': prices_log_scaled[:a_end],
            'test': prices_log_scaled[a_end:b_end]
        }
    ]
    
    # 保存原始时序图
    plot_and_save_timeseries(dates, prices_log_scaled, 
                            '标准化对数收盘价时间序列', 
                            stock_code, 
                            plot_type="original")
    
    print("2. 进行平稳性检验...")
    d = 0
    curr_data = prices_log_scaled.copy()
    curr_dates = dates
    
    while d < 2:
        if check_stationarity_all(curr_data):
            break
        print(f"   执行{d+1}阶差分...")
        d += 1
        curr_data = np.diff(curr_data)
        curr_dates = dates[d:]
        plot_and_save_timeseries(curr_dates, curr_data,
                                f'{d}阶差分后的时间序列',
                                stock_code,
                                plot_type=f"diff_{d}")
    
    print(f"3. 开始{rounds}轮训练...")
    round_metrics = []
    
    # 预先计算差分数据
    diff_data = []
    curr_data = prices_log_scaled.copy()
    for i in range(2):
        if check_stationarity_all(curr_data):
            break
        curr_data = np.diff(curr_data)
        diff_data.append(curr_data)
    d = len(diff_data)
    
    # 预先计算所有轮次的训练和测试数据
    train_test_data = []
    for round_num in range(rounds):
        train_data = rounds_data[round_num]['train']
        test_data = rounds_data[round_num]['test']
        if d > 0:
            train_data = np.diff(train_data, n=d)
            test_data = np.diff(test_data, n=d)
        train_test_data.append((train_data, test_data))
    
    # 使用预计算的数据进行训练
    for round_num, (train_data, test_data) in enumerate(train_test_data):
        print(f"\n   第{round_num+1}轮:")
        print("   - 数据划分...")

        # 获取对应的日期范围，注意考虑差分导致的长度变化
        if round_num == 0:
            test_dates = dates[d_end+d:e_end]  # 加上d以补偿差分
        elif round_num == 1:
            test_dates = dates[e_end+d:a_end]
        else:
            test_dates = dates[a_end+d:b_end]
            
        print("   - 网格搜索最优参数...")
        best_order = grid_search_parameters(train_data, d)
        print(f"   - 最优参数: p={best_order[0]}, d={best_order[1]}, q={best_order[2]}")
        
        print("   - 模型训练与预测...")
        # 训练模型并预测
        model = ARIMA(train_data, order=best_order)
        results = model.fit()
        
        # 逐步预测
        forecast = []
        history = list(train_data)
        for t in range(len(test_data)):
            model = ARIMA(history, order=best_order)
            model_fit = model.fit()
            yhat = model_fit.forecast(steps=1)[0]
            forecast.append(yhat)
            history.append(test_data[t])
        forecast = np.array(forecast)
        
        print("   - 计算性能指标...")
        # 计算性能指标
        rae = np.sum(np.abs(test_data - forecast)) / np.sum(np.abs(test_data - np.mean(test_data)))
        rse = np.sum((test_data - forecast)**2) / np.sum((test_data - np.mean(test_data))**2)
        round_metrics.append({'RAE': rae, 'RSE': rse})
        
        # 使用新的绘图函数
        plot_prediction_results(test_dates, test_data, forecast, stock_code, round_num)
    
    # 计算平均性能指标
    avg_rae = np.mean([m['RAE'] for m in round_metrics])
    avg_rse = np.mean([m['RSE'] for m in round_metrics])
    
    plt.close('all')
    gc.collect()
    
    return {
        'stock_code': stock_code,
        'avg_rae': avg_rae,
        'avg_rse': avg_rse
    }

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
    
    # 并行处理多个股票
    num_cores = mp.cpu_count() // 4
    with Parallel(n_jobs=num_cores) as parallel:
        results = parallel(
            delayed(process_single_stock_optimized)(file) 
            for file in tqdm(excel_files, desc="总体进度")
        )
    
    # 处理结果
    all_results = []
    for result in results:
        if result:  # 检查是否成功处理
            all_results.append({
                '股票文件': result['stock_code'],
                '平均RAE(三轮)': result['avg_rae'],
                '平均RSE(三轮)': result['avg_rse']
            })
    
    print("\n所有股票处理完成!")
    # 保存所有结果
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("所有股票ARIMA结果.csv", index=False)
    print("已保存结果到: 所有股票ARIMA结果.csv")

if __name__ == "__main__":
    # 设置matplotlib后端
    matplotlib.use('Agg')    
    # 忽略警告
    warnings.filterwarnings("ignore")
    main()
