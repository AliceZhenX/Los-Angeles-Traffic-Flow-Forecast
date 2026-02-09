import pandas as pd
import numpy as np
import os

H5_PATH = "data/metr-la.h5"
STRICT_MODE = True  # True: 剔除故障样本; False: 保留故障样本
OUTPUT_DIR = "clean_data" if STRICT_MODE else "clean_data_with_failures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 12
HORIZON = 12

def get_bad_sensors_from_train(df_train, threshold=0.1):
    """仅从训练集识别坏传感器"""
    zero_rates = (df_train == 0).mean(axis=0)
    return zero_rates[zero_rates > threshold].index.tolist()

# def refined_spatial_impute(df, bad_sensors):
#     """仅修复黑名单传感器的 0 值点"""
#     df_clean = df.copy()
#     good_sensors = df.columns.difference(bad_sensors)
#     network_mean = df[good_sensors].mean(axis=1)
#     for sensor in bad_sensors:
#         zero_mask = (df_clean[sensor] == 0)
#         df_clean.loc[zero_mask, sensor] = network_mean[zero_mask]
#     return df_clean

def refined_spatial_impute(df, bad_sensors):
    """
    1. 修复黑名单节点的所有 0 值。
    2. 修复健康节点中零星出现的 0 值噪声。
    """
    df_clean = df.copy()
    
    # 获取全网健康节点
    good_sensors = df.columns.difference(bad_sensors)
    network_mean = df[good_sensors].mean(axis=1)
    
    # 遍历所有传感器 (207个)
    for sensor in df.columns:
        # 找到该传感器读数为 0 的位置
        zero_mask = (df_clean[sensor] == 0)
        
        # 只有当全网均值 > 0 时才修复,避免在全网系统瘫痪时刻（为0）填入无效数据
        actual_fix_mask = zero_mask & (network_mean > 0)
        
        df_clean.loc[actual_fix_mask, sensor] = network_mean[actual_fix_mask]
        
    print(f"全局 0 值噪声修复完成。")
    return df_clean

def generate_samples(df, x_offsets, y_offsets, split_name, is_strict):
    """修正边界逻辑的样本生成"""
    num_samples, num_nodes = df.shape
    data_val = df.values
    
    # 时间特征
    time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    time_in_day = np.tile(time_ind, [num_nodes, 1]).transpose((1, 0))
    data_combined = np.stack([data_val, time_in_day], axis=-1)

    x, y = [], []
    skipped = 0
    
    # 确保 t + max(y_offsets) 不越界
    min_t = abs(min(x_offsets))
    max_t = num_samples - abs(max(y_offsets))
    
    for t in range(min_t, max_t): # 注意这里直接用 max_t
        # 故障检测
        x_window = data_val[t + x_offsets, :]
        is_failure = np.any((x_window == 0).mean(axis=1) > 0.5)
        
        if is_strict and is_failure:
            skipped += 1
            continue
            
        x.append(data_combined[t + x_offsets, ...])
        y.append(data_combined[t + y_offsets, ...])

    print(f"[{split_name}] Mode: {'Strict' if is_strict else 'Full'}, Samples: {len(x)}, Skipped: {skipped}")
    return np.stack(x), np.stack(y)

def main():
    df = pd.read_hdf(H5_PATH)
    train_end = int(len(df) * 0.7)
    val_end = int(len(df) * 0.8)
    
    # 1. 划分与空间修复
    df_train = df.iloc[:train_end]
    bad_sensors = get_bad_sensors_from_train(df_train)
    
    # 2. 对所有分集执行局部空间修复
    df_train_cl = refined_spatial_impute(df_train, bad_sensors)
    df_val_cl = refined_spatial_impute(df.iloc[train_end:val_end], bad_sensors)
    df_test_cl = refined_spatial_impute(df.iloc[val_end:], bad_sensors)
    
    # 3. 训练集时间维插值
    time_zero_rates = (df_train_cl == 0).mean(axis=1)
    bad_times = time_zero_rates[time_zero_rates > 0.5].index
    df_train_cl.loc[bad_times, :] = np.nan
    df_train_cl = df_train_cl.interpolate(method='linear', axis=0).ffill().bfill()

    # 4. 生成数据
    x_offsets = np.sort(np.arange(-SEQ_LEN + 1, 1))
    y_offsets = np.sort(np.arange(1, HORIZON + 1))
    
    x_train, y_train = generate_samples(df_train_cl, x_offsets, y_offsets, "Train", is_strict=False)
    x_val, y_val = generate_samples(df_val_cl, x_offsets, y_offsets, "Val", is_strict=STRICT_MODE)
    x_test, y_test = generate_samples(df_test_cl, x_offsets, y_offsets, "Test", is_strict=STRICT_MODE)

    np.savez_compressed(f"{OUTPUT_DIR}/train.npz", x=x_train, y=y_train)
    np.savez_compressed(f"{OUTPUT_DIR}/val.npz", x=x_val, y=y_val)
    np.savez_compressed(f"{OUTPUT_DIR}/test.npz", x=x_test, y=y_test)
    print(f"Dataset saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()