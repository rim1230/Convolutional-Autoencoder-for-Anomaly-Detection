import os
import pandas as pd

# ディレクトリのパスを指定
directory_path = '/mnt/storage/TDR_data/diff_sampled/kyoto/min/'

# CSVファイルの第4列のデータを格納するリスト
all_data = []

# ディレクトリ内のすべてのCSVファイルを読み込む
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)
        
        # CSVを読み込み、第4列を取り出す
        df = pd.read_csv(file_path)
        
        # 第4列のデータをリストに追加
        all_data.append(df.iloc[:, 1])  # ilocで第4列を指定（0始まりのため3）

# 各時間の平均を計算（入力例で確認済み）
df_all = pd.concat(all_data, axis=1)
average_series = df_all.mean(axis=1)

average_series.to_csv('~/tdr_ae/results/analysis/average_min.csv', index=False, header=False)
