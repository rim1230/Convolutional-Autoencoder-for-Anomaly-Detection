import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

def plot_wave(file_name):
    """
    コマンドライン引数で指定したファイルの2列目をプロットする
    """

    # CSVファイルを読み込む
    data = pd.read_csv(file_name)  # header=Noneでヘッダー行がないことを明示
    vol = data.iloc[:,0]  # 0から始まるインデックスで2行目は1
    
    # プロット
    plt.figure(figsize=(10, 5))
    t = np.linspace(0, len(vol)-1, len(vol))
    plt.plot(t, vol, label='Voltage')
    plt.xlabel("Index")
    plt.ylabel("Voltage")
    # plt.title(f"Voltage of {file_name}")
    plt.legend()
    plt.grid()
    plt.show()
    # plt.savefig('{file_name}.png')

if __name__ == "__main__":
    # コマンドライン引数からファイル名を取得
    file_name = sys.argv[1]
    plot_wave(file_name)
