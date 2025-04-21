import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm

def compute_def_waveform(input_dir_path):
    # 入力ディレクトリ名を取得
    dir_name = os.path.basename(input_dir_path.rstrip("/"))

    # 入力ディレクトリ内のすべてのCSVファイルを取得
    files = [f for f in os.listdir(input_dir_path) if f.endswith('.csv')]
    files = sorted(files)
    pd.DataFrame(files)
    # print(files)

    # # DataFrameに変換（各要素を行として保存）
    # df = pd.DataFrame(files, columns=["File Names"])

    # # CSVとして保存(filesの順序ファイル)
    # df.to_csv(f"{dir_name}output_files.csv", index=False)

    # データを格納するリスト
    data = []

    for file in files:
        input_file_path = os.path.join(input_dir_path, file)
        
        # 入力CSVをロード
        df = pd.read_csv(input_file_path, header=None)
        
        # 差分と正規化
        # float32のNumpy配列
        mode2 = (df.iloc[:, 0].to_numpy() + df.iloc[:, 1].to_numpy())
        max_val = mode2.max()
        min_val = mode2.min()
        normalized_mode2 = (2 * (mode2 - min_val) / (max_val - min_val) - 1).astype(np.float32)

        # Numpy配列をリストにappendして、配列自体をリストの要素として追加
        data.append(normalized_mode2)
        # print(normalized_mode2.dtype)  # 正しく float32 になっているか確認

    # [19999, 360]
    data = np.array(data)
    # print(data.shape)

    output_file_name = f"{output_base_dir}/{dir_name}.csv"
    np.savetxt(output_file_name, data, delimiter=',', fmt='%.6f')


# 出力ディレクトリの設定
output_base_dir = '/mnt/storage/TDR_data/diff/kyoto/all-2'
os.makedirs(output_base_dir, exist_ok=True)  # 出力ディレクトリを作成（存在しない場合のみ）

# カレントディレクトリを取得
current_dir = os.getcwd()

# 数字から始まるディレクトリのみを取得
directories = [d for d in os.listdir(current_dir) 
               if os.path.isdir(os.path.join(current_dir, d)) and d[0].isdigit() and int(d) > 201901270345]

# 昇順にソート
directories.sort()
# print(directories)

# 各ディレクトリに処理を適用
for dir in tqdm(directories, desc="Processing Directories"):
    compute_def_waveform(dir)
