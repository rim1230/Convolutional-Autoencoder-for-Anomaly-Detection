import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_data(path):
    # CSVファイルのパス
    csv_file_path = path
    # CSVをDataFrameに読み込む
    df = pd.read_csv(csv_file_path, header=None)  # header=None でヘッダーを無視

    plt.figure(figsize=(10, 6))  # プロットのサイズを指定

    # 平均波形をロード
    df2 = pd.read_csv(r'~/tdr_ae/results/analysis/average.csv', header=None)
    plt.plot(df2, color='black', zorder=361, label='Average')

    # 各行（時系列データ）をプロット
    for i in range(df.shape[0]):  # 行ごとにプロット
        plt.plot(df.iloc[i], label=f'Series {i+1}')


    # グラフの設定
    plt.grid()
    plt.title(f'{path}')
    plt.xlabel('Time')
    plt.ylabel('Mode 2 Voltage')
    plt.legend(loc='upper right')  # 凡例を表示
    plt.show()

if __name__ == '__main__':
    args = sys.argv
    plot_data(args[1])