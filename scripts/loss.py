import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import time

# GPUを利用する場合 (CuPy)
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# データの読み込みとMAE計算
def calculate_loss(data_file, average_data):
    # データの読み込み
    data = pd.read_csv(data_file, header=None).values  # N行M列
    if GPU_AVAILABLE:
        data = cp.array(data)
        average_data = cp.array(average_data)

    # MAE計算
    mae = np.mean(np.abs(data - average_data), axis=1) if not GPU_AVAILABLE else cp.mean(cp.abs(data - average_data), axis=1)
    return cp.asnumpy(mae) if GPU_AVAILABLE else mae

start = time.perf_counter() #計測開始
###

# 平均化データをロード
average = pd.read_csv('results/analysis/average.csv', header=None).values.flatten()

# 処理するディレクトリ内のCSVファイル一覧 (昇順)
data_files = sorted(glob.glob('data/test/*.csv'))

# MAEをまとめるリスト
all_mae = []

# 各ファイルを順に処理
for idx, data_file in enumerate(tqdm(data_files, desc="Processing files")):
    # MAE計算
    mae = calculate_loss(data_file, average)
    
    all_mae.append(mae)

all_mae = all_mae.flatten()

# すべてのMAEを1つのデータフレームにまとめる
all_mae_df = pd.DataFrame(all_mae)

# 結果を1つのCSVファイルに保存
all_mae_df.to_csv('results/all_mae_results.csv', header=False, index=False)

###
end = time.perf_counter() #計測終了
print('{:.2f}'.format(end - start)) # 秒で計測時間を表示
