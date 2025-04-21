import os
import sys
import itertools
import concurrent.futures

# ディレクトリと出力ファイルのパス
directory = "./"
output_file = "merged.csv"

# ファイル名を昇順でソート
files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv")])

# 一時ファイルの保存ディレクトリ
temp_dir = "temp_merge"
os.makedirs(temp_dir, exist_ok=True)

def process_chunk(chunk_files, chunk_id):
    """ファイルチャンクを結合して一時ファイルに保存"""
    temp_file = os.path.join(temp_dir, f"temp_{chunk_id}.csv")
    with open(temp_file, 'w') as outfile:
        for file in chunk_files:
            with open(file, 'r') as infile:
                for line in infile:
                    outfile.write(line)
    return temp_file

# 並列でファイルを処理
num_workers = 4  # 並列処理のワーカー数（環境に応じて調整）
chunk_size = len(files) // num_workers + 1
chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]

with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    temp_files = list(executor.map(process_chunk, chunks, range(len(chunks))))

# 一時ファイルを結合
with open(output_file, 'w') as outfile:
    for temp_file in temp_files:
        with open(temp_file, 'r') as infile:
            for line in infile:
                outfile.write(line)

# 一時ファイルを削除
for temp_file in temp_files:
    os.remove(temp_file)
os.rmdir(temp_dir)

print(f"結合完了！結果は {output_file} に保存されました。")
