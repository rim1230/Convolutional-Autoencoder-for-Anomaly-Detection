import os
import pandas as pd
import h5py

# CSVファイルがあるディレクトリのパス
csv_dir = 'data/test'

# HDF5ファイルの出力先
hdf5_file = 'data/test/data.h5'

# ディレクトリ内のCSVファイルを時系列順（昇順）にソート
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
csv_files.sort()  # ファイル名で昇順ソート（ファイル名が日付順であることを前提）

# HDF5形式でデータを書き込む
with h5py.File(hdf5_file, 'w') as hf:
    for filename in csv_files:
        csv_path = os.path.join(csv_dir, filename)
        df = pd.read_csv(csv_path)
        
        # HDF5ファイル内にCSVの内容を保存
        # CSVファイル名（拡張子なし）をHDF5のグループ名として使用
        group_name = os.path.splitext(filename)[0]
        
        # データをデータセットとして追加
        hf.create_dataset(group_name, data=df.values)

        print(f"Converted {filename} to HDF5 format.")