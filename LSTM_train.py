import os
import glob
import torch
import numpy as np
import pandas as pd
from torch import optim
from LSTM import (
    clean_stock_data,
    add_technical_features,
    prepare_data_forward_link,
    create_dataloader,
    LSTMModel,
    train_lstm,
    evaluate_lstm,
    calculate_metrics,
)
import matplotlib.pyplot as plt
# 设置中文显示
plt.rcParams['font.family'] = ['Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False

def plot_training_history(history, title="模型训练历史"):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label='训练损失')
    plt.plot(history['val_losses'], label='验证损失')  
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

def plot_predictions(dates, true_values, pred_values, title="预测结果对比"):
    """绘制预测结果对比图"""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, true_values, label='真实值')
    plt.plot(dates, pred_values, label='预测值')
    plt.title(title)
    plt.xlabel('日期')
    plt.ylabel('标准化股价')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

def evaluate_and_plot(model, test_loader, scaler, dates, fold_index, stock_code, save_dir):
    """评估模型并绘制预测结果"""
    # 1. 获取预测值和真实值
    preds, trues = evaluate_lstm(model, test_loader, scaler)
    
    # 根据轮次选择对应的日期段
    if fold_index == 0:
        # 第一轮: f段日期
        test_dates = dates[int(len(dates)*0.7*0.9):int(len(dates)*0.7)]
    elif fold_index == 1:
        # 第二轮: b段日期
        test_dates = dates[int(len(dates)*0.7):int(len(dates)*0.9)]
    else:
        # 第三轮: c段日期
        test_dates = dates[int(len(dates)*0.9):]
        
    plot_predictions(
        dates=test_dates,  # 使用对应轮次的日期段
        true_values=trues,
        pred_values=preds,
        title=f"{stock_code} 第{fold_index+1}轮测试集预测结果"
    )
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存图片
    save_path = os.path.join(save_dir, f"{stock_code}_第{fold_index+1}轮测试集预测结果.png")
    plt.savefig(save_path)
    plt.close()  # 立即关闭图形
    
    return preds, trues

def main():
    # 1. 指定股票数据所在的文件夹路径
    folder_path = "每日行情/"  
    # 2. 搜索指定文件夹下以 .xlsx 结尾的文件(多支股票)
    excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
    
    # 若未找到任何Excel文件，则退出
    if not excel_files:
        print("未找到任何Excel文件，请检查文件夹路径。")
        return
    
    # 用于汇总所有股票的评估结果
    all_results = []

    # 3. 遍历每日行情文件夹中的所有Excel文件
    for file in excel_files:
        # ---(A) 数据读取与清洗部分---
        
        # 读取Excel文件后，将日期列转为Datetime并设为索引
        df = pd.read_excel(file, skiprows=2, nrows=1690)
        df['交易日期'] = pd.to_datetime(df['交易日期'])
        df.set_index('交易日期', inplace=True)
        df = clean_stock_data(df)

        # ---(B) 特征工程部分---
        
        # 添加技术指标，形成共10个特征
        df = add_technical_features(df, window=7)
        feature_cols = [
            '开盘价(元)', '最高价(元)', '最低价(元)', '成交量(股)', '成交额(元)',
            'STD', 'EWMA', 'MOM', 'RSI', 'WR'
        ]
        target_col = '收盘价(元)'

        # ---(C) 数据集前向链接法划分---
        
        # 调用prepare_data_forward_link，只返回三轮 (train, val, test) 切分
        data_dict = prepare_data_forward_link(df, feature_cols, target_col)
        forward_folds = data_dict['forward_folds']
        scaler = data_dict['scaler']

        # ---(D) 三轮的训练与测试---
        
        fold_metrics = []  # 记录三次测试集的误差指标
        for i, fold in enumerate(forward_folds):
            train_X, train_y = fold['train_X'], fold['train_y']
            val_X,   val_y   = fold['val_X'],   fold['val_y']
            test_X,  test_y  = fold['test_X'],  fold['test_y']
            
            # [注] LSTM期望输入维度为 (batch_size, seq_len, feature_dim)
            # 这里简单扩展到三维: (样本数, 1, 特征数)
            train_X = np.expand_dims(train_X, axis=1)
            val_X   = np.expand_dims(val_X,   axis=1)
            test_X  = np.expand_dims(test_X,  axis=1)
            
            # 可选: 打印本轮训练数据形状
            print(f"第{i+1}轮: Train X shape = {train_X.shape}, Train y shape = {train_y.shape}")

            # 构建DataLoader
            train_loader = create_dataloader(train_X, train_y, batch_size=64)  # 增大batch_size
            val_loader = create_dataloader(val_X, val_y, batch_size=64)
            test_loader = create_dataloader(test_X, test_y, batch_size=64)

            # 初始化增强版LSTM模型
            input_size = train_X.shape[2]
            model = LSTMModel(
                input_size=input_size,
                hidden_size=128,      # 增大hidden_size
                num_layers=3,         # 增加层数
                output_size=1,
                dropout=0.2          # 添加dropout
            )
            
            # 使用改进的优化器设置
            criterion = torch.nn.MSELoss()
            optimizer = optim.AdamW(  # 使用AdamW优化器
                model.parameters(),
                lr=0.001,
                weight_decay=0.01    # L2正则化
            )
            
            # 训练模型并获取历史记录
            model, history = train_lstm(
                train_loader,
                val_loader, 
                model,
                criterion,
                optimizer,
                num_epochs=100        # 增加训练轮数
            )
            
            stock_code = os.path.basename(file)  # 例如: '600519.SH.csv' 
            stock_code = stock_code.rsplit('.', 1)[0]  # 去掉 .csv，保留 '600519.SH'

            # 创建保存目录
            save_dir = "LSTM训练过程"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 绘制训练过程
            plot_training_history(history, f"{stock_code} 第{i+1}轮训练历史")
            # 保存训练过程图像
            plt.savefig(os.path.join(save_dir, f"{stock_code}_第{i+1}轮训练历史.png"))
            plt.close()

            # 评估并绘制预测结果
            # preds, trues = evaluate_lstm(model, test_loader, scaler)
            # plot_predictions(
            #     dates=df.index[-len(preds):],
            #     true_values=trues,
            #     pred_values=preds,
            #     title=f"{stock_code} 第{i+1}轮测试集预测结果"
            # )
            preds, trues = evaluate_and_plot(
                model, test_loader, scaler,
                df.index, i, stock_code, save_dir
            )            
            # 保存预测结果图像
            # plt.savefig(os.path.join(save_dir, f"{stock_code}_第{i+1}轮测试集预测结果.png"))
            # plt.close()
            
            # 计算评估指标
            rae, rse = calculate_metrics(trues, preds)
            fold_metrics.append((rae, rse))
            print(f"第{i+1}轮: Test集RAE={rae:.4f}, RSE={rse:.4f}")

        # 统计三轮测试集指标的平均值
        if fold_metrics:
            avg_rae = np.mean([m[0] for m in fold_metrics])
            avg_rse = np.mean([m[1] for m in fold_metrics])
        else:
            avg_rae, avg_rse = np.nan, np.nan

        # 将结果保存到all_results
        file_name = os.path.basename(file)
        all_results.append({
            '股票文件': file_name,
            '平均RAE(三轮)': avg_rae,
            '平均RSE(三轮)': avg_rse
        })
        
        print(f"\n处理完成: {file_name}, 三轮平均 RAE={avg_rae:.4f}, RSE={avg_rse:.4f}\n")
    
    # ---(E) 汇总所有股票的结果并保存---
    results_df = pd.DataFrame(all_results)
    print("所有股票模型表现:")
    print(results_df)
    results_df.to_csv("所有股票LSTM结果.csv", index=False)
    print("\n已保存结果到: 所有股票LSTM结果.csv")

if __name__ == "__main__":
    main()