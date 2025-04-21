import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

def plot_wave(file_name):
    """
    コマンドライン引数で指定したファイルの2列目をプロットする
    """

    # 異常ファイルを読み込む
    data_1 = pd.read_csv(file_name)  # プログラム実行時のコマンドライン引数で指定した異常なファイルをロード
    anomaly = data_1.iloc[:,3]  # csvファイルの何列目かを指定 
    # 正常ファイルを読み込む
    healthy_file_name = '20190121121224722.csv'
    data_2 = pd.read_csv(healthy_file_name) # 適当なファイルを正常ファイルとしてロード
    healthy = data_2.iloc[:,3]  # 0から始まるインデックスで2行目は1
    
    # プロット
    plt.figure()
    t = np.linspace(0, len(anomaly)-2, len(anomaly))
    plt.plot(t, anomaly, c='tab:orange', alpha=0.7, label=f'{file_name}')
    plt.plot(t, healthy, c='tab:blue', label=f'{healthy_file_name}')
    # plt.scatter(t, healthy, alpha=0.1, s=10, label='Healthy')
    # plt.xlim([2600, 3000])
    plt.xlabel("Time [ns]")
    plt.ylabel("Mode 2 Voltage [V]")
    plt.title(f"Voltage of {file_name}")
    plt.legend()
    plt.grid()
    plt.show()
    # plt.savefig('{file_name}.png')

if __name__ == "__main__":
    # コマンドライン引数からファイル名を取得
    file_name = sys.argv[1]
    plot_wave(file_name)
