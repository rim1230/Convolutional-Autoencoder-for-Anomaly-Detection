import os
import cupy as cp
import pandas as pd
from tqdm import tqdm

# 出力ディレクトリの指定
output_dir = "/mnt/storage/TDR_data/diff_sampled/kyoto/all/"

# 出力ディレクトリを作成（存在しない場合）
os.makedirs(output_dir, exist_ok=True)

# リサンプリング関数（CuPyを使用）
def resample_csv(input_file, output_file):
    try:
        # CSVをPandasで読み込み、NumPy配列として取得
        data = pd.read_csv(input_file, header=None).values
        # NumPy配列をCuPy配列に変換してGPUに転送
        data_gpu = cp.asarray(data)
        # リサンプリング（列を間引き、間隔は10）
        resampled_data_gpu = data_gpu[:, ::10]
        # CuPy配列をNumPy配列に戻してCPUに転送
        resampled_data = cp.asnumpy(resampled_data_gpu)
        # NumPy配列をDataFrameに変換して保存
        resampled_df = pd.DataFrame(resampled_data)
        resampled_df.to_csv(output_file, index=False, header=False)
    except Exception as e:
        print(f"エラーが発生しました: {input_file}, {e}")

# カレントディレクトリ以下のCSVファイルを探索
all_csv_files = []
for root, _, files in os.walk("."):
    for file in files:
        if file.endswith(".csv"):
            all_csv_files.append(os.path.join(root, file))

# ファイル名の昇順でソート
all_csv_files = sorted(all_csv_files)

# tqdmを使って進捗を表示
with tqdm(total=len(all_csv_files), desc="Processing CSV files") as pbar:
    # 各ファイルを順番に処理
    for input_file in all_csv_files:
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        resample_csv(input_file, output_file)
        pbar.update(1)

print("すべてのCSVファイルのリサンプリングが完了しました。")
