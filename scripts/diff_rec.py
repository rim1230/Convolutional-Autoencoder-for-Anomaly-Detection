import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm

# 出力ディレクトリの設定
output_base_dir = '/mnt/storage/TDR_data/diff/kyoto/w_core_500m/'
os.makedirs(output_base_dir, exist_ok=True)  # 出力ディレクトリを作成（存在しない場合のみ）

def compute_def_waveform(input_file_path, output_base_dir):
    # 入力CSVをロード
    df = pd.read_csv(input_file_path, header=None)

    # 差分と正規化
    df["diff"] = (df.iloc[:, 0] + df.iloc[:, 1])
    df["diff"] = 2 * (df["diff"] - df["diff"].min()) / (df["diff"].max() - df["diff"].min()) - 1

    # 出力ファイルパスの作成（元ファイル名を保持）
    output_file_name = os.path.basename(input_file_path)
    output_file_path = os.path.join(output_base_dir, output_file_name)

    # CSVを出力
    df.to_csv(output_file_path)  # インデックス不要なら`index=False`

# 最初のCSVファイルのパスを取得
paths = []
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".csv"):
            paths.append(os.path.join(root, file))

# 各ファイルに処理を適用
for input_file_path in tqdm(paths, desc="Processing files"):
    compute_def_waveform(input_file_path, output_base_dir)
