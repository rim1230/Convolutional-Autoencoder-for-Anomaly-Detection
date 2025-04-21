import pandas as pd
import matplotlib.pyplot as plt
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 処理するディレクトリ内のCSVファイル一覧を取得 (昇順)
data_files = sorted(glob.glob('/home/takayanagi/data/diff/kyoto/all/*.csv'))

# ファイルの行数を取得する関数
def get_row_count(file):
    with open(file, 'r') as f:
        return sum(1 for _ in f)

# 並列処理で行数を取得
row_counts = []
with ThreadPoolExecutor() as executor:
    row_counts = list(tqdm(executor.map(get_row_count, data_files), total=len(data_files), desc="Counting rows"))

# プロット
plt.figure(figsize=(10, 6))
plt.plot(range(len(data_files)), row_counts)
plt.title("Number of Rows in CSV Files")
plt.xlabel("File Index (Sorted Order)")
plt.ylabel("Number of Rows")
plt.grid(True)
plt.tight_layout()
plt.savefig("row_counts_plot.png")  # 保存

# row_countsを出力
# pandas DataFrameに変換
df = pd.DataFrame(row_counts)

# CSVファイルに出力（ヘッダーとインデックスなし）
output_file = "row_counts.csv"
df.to_csv(output_file, index=False, header=False)
