import os
import math
import pandas as pd

# 対象ディレクトリ
dir_path = "./"

# CSVファイルを昇順に取得
csv_files = sorted([f for f in os.listdir(dir_path)
                    if f.endswith(".csv") and os.path.isfile(os.path.join(dir_path, f))])

group_size = 24
num_groups = math.ceil(len(csv_files) / group_size)

for i in range(num_groups):
    group = csv_files[i*group_size : (i+1)*group_size]
    df = pd.read_csv(os.path.join(dir_path, group[0]), header=None, skiprows=0)
    print(df)
    dfs = [pd.read_csv(os.path.join(dir_path, gf), header=None, skiprows=0) for gf in group]
    print(dfs)
    merged_df = pd.concat(dfs, ignore_index=True)
    print(merged_df)
    merged_df.to_csv(os.path.join(dir_path, f"merged/01{i+18:02d}.csv"), header=False, index=False)