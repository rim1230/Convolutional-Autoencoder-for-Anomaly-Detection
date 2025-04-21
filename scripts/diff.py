import numpy as np
import pandas as pd
import glob
import os

def compute_def_waveform(file_path):
    # load CSV
    df = pd.read_csv(file_path, header=None)

    # 差分と正規化
    df["diff"] = (df.iloc[:, 0] + df.iloc[:, 1]) / 2
    df["diff"] = 2 * (df["diff"] - df["diff"].min()) / (df["diff"].max() - df["diff"].min()) - 1
    output_file_path = f'../../../diff/kyoto/1hour/{file_path}'
    df.to_csv(output_file_path)

# ロードするデータのパス名のリストを取得
# 
file_paths = glob.glob('*.csv')

for file_path in file_paths:
    compute_def_waveform(file_path)