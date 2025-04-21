import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def median1d(arr, k):
    w = len(arr)
    idx = np.fromfunction(lambda i, j: i + j, (k, w), dtype=int) - k // 2
    idx[idx < 0] = 0
    idx[idx > w - 1] = w - 1
    return np.median(arr[idx], axis=0)

# 処理するデータのディレクトリ
directory = '/home/takayanagi/data/diff/kyoto/wo_core/'

def process_file(filename):
    if filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)

        # データの読み込み（空白や不正値を NaN として扱う）
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)

        # 欠損値 (NaN) を補完する処理
        # 線形補完
        for i in range(data.shape[1]):
            column = data[:, i]
            nan_indices = np.isnan(column)
            if np.any(nan_indices):  # NaN が存在する場合のみ補完
                valid_indices = np.where(~nan_indices)[0]
                valid_values = column[valid_indices]
                data[nan_indices, i] = np.interp(np.where(nan_indices)[0], valid_indices, valid_values)

        # メディアンフィルタを適用
        filtered_data = np.apply_along_axis(median1d, 1, data, 5)

        # 保存先のディレクトリ
        save_dir = '/home/takayanagi/data/diff_clean/kyoto/wo_core/'
        os.makedirs(save_dir, exist_ok=True)  # 出力ディレクトリを作成（存在しない場合のみ）
        save_path = os.path.join(save_dir, filename)
        np.savetxt(save_path, filtered_data, delimiter=',', fmt='%.6f')

with ProcessPoolExecutor() as executor:
    filenames = [f for f in os.listdir(directory) if f.endswith('.csv')]
    list(tqdm(executor.map(process_file, filenames), total=len(filenames), desc="Processing"))