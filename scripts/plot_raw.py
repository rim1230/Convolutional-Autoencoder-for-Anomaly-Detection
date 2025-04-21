import os
import pandas as pd
import matplotlib.pyplot as plt

# 現在のディレクトリ名を取得
current_dir_name = os.path.basename(os.getcwd())

# 現在のディレクトリ内のすべてのCSVファイルを取得
csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]
csv_files = sorted(csv_files)

# プロットしたいファイルをファイル名で指定する
target_file = '20190128091600085.csv'
before_n = 100
# 前に取得するファイルの数
after_n = 0 # 後に取得するファイルの数
index = csv_files.index(target_file)
before_files = csv_files[max(0, index - before_n):index]
after_files = csv_files[index + 1: index + 1 + after_n]

plt.figure()

# 各CSVファイルをプロット
for csv_file in csv_files:
    # プロットしたいCSVファイル名をif文で指定
    # すべてをプロットしたい場合はコメントアウトを操作する
    # if True:
    if csv_file == target_file or csv_file in before_files or csv_file in after_files:
        # CSVファイルを読み込む
        df = pd.read_csv(csv_file, header=None)
        df['diff'] = (df.iloc[:,0] + df.iloc[:,1]) / 2
        data = 2 * (df["diff"] - df["diff"].min()) / (df["diff"].max() - df["diff"].min()) - 1

        if csv_file == target_file:
            # zorder: プロットの重ね順（大きいと前面）
            plt.plot(data, c='red', alpha=0.5, zorder=365, label=csv_file)
        else:
            plt.plot(data, c='black', alpha=0.2, label=csv_file)
    else:
        continue

plt.xlabel("Time [ns]")
plt.ylabel("Mode 2 Voltage [V]")
plt.title(current_dir_name)
# plt.legend()
plt.grid()
plt.show()