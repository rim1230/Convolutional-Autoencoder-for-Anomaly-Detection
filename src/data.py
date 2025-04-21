import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import csv
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import random
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = json.load(f)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

seed = config["seed"]

g = torch.Generator()
g.manual_seed(seed)

class Data:
    def __init__(self, batch_size, test_batch_size, val_split, shuffle=True):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.val_split = val_split
        self.shuffle = shuffle
        self.current_index = 432 - 60 # 学習データの開始インデックス

    def make_file_list(self, directory):
        file_list = os.listdir(directory)
        file_list_sorted = sorted(file_list) # 日付順にソート
        return file_list_sorted

    def load_data_from_directory(self, directory, file_list, column_index, skip_rows, nrows):
        """
        ディレクトリ内のCSVファイルを読み込み、テンソルを作成する
        :param directory: CSVファイルが格納されたディレクトリのパス
        :param column_index: 抽出する列のインデックス（0ベース）
        :param skip_rows: スキップする行数
        :return: データを格納したテンソル (torch.Tensor)
        """
        def load_csv(file_path):
            data = pd.read_csv(file_path, usecols=[column_index], header=None)
            return data.values.flatten()

        # ディレクトリ内のCSVファイルを並列処理で読み込む
        with ThreadPoolExecutor() as executor:
            x_train = list(tqdm(executor.map(load_csv, [os.path.join(directory, filename) for filename in file_list if filename.endswith(".csv")]), total=len(file_list), desc="Loading CSV files"))

        # NumPy配列に変換してテンソル化
        x_train = np.array(x_train)
        x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)  # [B, 1, T]
        return x_train

    def prepare_dataset(self, x_data, y_data=None):
        """
        データセットとデータローダーを作成する
        :param x_data: 入力データ (torch.Tensor)
        :param y_data: ラベルデータ (torch.Tensor, optional)。Noneの場合、x_dataをそのまま使用。
        :return: 訓練用データローダーと検証用データローダー
        """
        if y_data is None:
            y_data = x_data

        dataset = TensorDataset(x_data, y_data)
        train_size = int((1 - self.val_split) * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=2, worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        print(f'Training batches: {len(train_loader)}')
        print(f'Validation batches: {len(val_loader)}')

        return train_loader, val_loader

    def load_csv_to_tensor(self, directory, file_list):
        """
        CSVファイルを読み込み、各行を1つの時系列データとして扱い、
        [B, 1, T]形式のテンソルに変換
        """
        tensors = []
        for file in file_list:
            file_path = os.path.join(directory, file)
            # CSVを読み込む (各行が時系列データ)
            data = np.loadtxt(file_path, delimiter=',', usecols=range(0, 5000))  # shape: (num_rows, num_cols)
            
            # 各行を時系列データとしてテンソル化
            tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # shape: [num_rows, 1, num_cols]
            
            tensors.append(tensor)
        
        # 複数のCSVからテンソルを結合して、[B, 1, T]の形にする
        return torch.cat(tensors, dim=0)  # shape: [B, 1, T]

    def next_batch(self, directory, file_list):
        """
        次のバッチを取得
        """
        if self.current_index >= len(file_list):
            return None  # データが終了
        
        # 次のバッチのファイルを取得
        batch_files = file_list[self.current_index:self.current_index + self.test_batch_size]
        self.current_index += self.test_batch_size

        # CSVを読み込んでテンソル化
        return self.load_csv_to_tensor(directory, batch_files)

    def prepare_test_dataset(self, x_data, batch_size, y_data=None): 
        """
        データをシャッフルせずにデータセットとデータローダーを作成する
        :param x_data: 入力データ (torch.Tensor)
        :param batch_size: バッチサイズ
        :param y_data: ラベルデータ (torch.Tensor, 任意)。Noneの場合、x_dataをそのまま使用。
        :return: テストデータローダー
        """
        if y_data is None:
            y_data = x_data

        dataset = TensorDataset(x_data, y_data)

        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        print(f'Test batches: {len(test_loader)}')

        return test_loader

    def plot_loss(self, train_loss, val_loss):
        """
        学習曲線をプロットする
        :param train_loss: 訓練誤差
        :param val_loss: 検証誤差
        """
        fig = plt.figure()
        plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
        plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()
        fig.savefig(os.path.join(config["output"]["dir"], "learning_curve.png"), dpi=300)


    def plot_two_waves(self, x_sample, reconstructed):
        """
        入力データと再構成データの波形をプロットする
        :param x_sample: 訓練データ
        :param reconstructed: x_sampleから再構成されたデータ
        """
        input_waveform = x_sample.cpu().numpy().flatten()
        reconstructed_waveform = reconstructed.cpu().numpy().flatten()

        t = np.linspace(0, len(input_waveform) - 1, len(input_waveform))
        fig = plt.figure(figsize=(14, 5))
        plt.plot(t, input_waveform, label='Input Waveform', alpha=0.7)
        plt.plot(t, reconstructed_waveform, label='Reconstructed Waveform', alpha=0.7)
        plt.legend()
        plt.title("Input vs Reconstructed Waveform")
        plt.ylim([-0.6, 0.6])
        plt.xlabel("Time [ns]")
        plt.ylabel("Voltage [V]")
        plt.show()

    def plot_four_waves(self, x_sample1, reconstructed1, x_sample2, reconstructed2):
        """
        入力データと再構成データの波形をプロットする
        :param x_sample: 訓練データ
        :param reconstructed: x_sampleから再構成されたデータ
        """
        input_waveform1 = x_sample1.cpu().numpy().flatten()
        input_waveform2 = x_sample2.cpu().numpy().flatten()
        reconstructed_waveform1 = reconstructed1.cpu().numpy().flatten()
        reconstructed_waveform2 = reconstructed2.cpu().numpy().flatten()

        t = np.linspace(0, len(input_waveform1) - 1, len(input_waveform1))
        fig = plt.figure(figsize=(14, 5))
        plt.plot(t, input_waveform1, label='Input Waveform 1', alpha=0.7)
        # plt.scatter(t, input_waveform1, alpha=0.7)
        plt.plot(t, input_waveform2, label='Input Waveform 2', alpha=0.7)
        # plt.scatter(t, input_waveform2, alpha=0.7)
        plt.plot(t, reconstructed_waveform1, zorder=6, label='Reconstructed Waveform 1', alpha=0.7)
        plt.plot(t, reconstructed_waveform2, zorder=5, label='Reconstructed Waveform 2', alpha=0.7)
        plt.legend()
        plt.title("Input vs Reconstructed Waveform")
        plt.xlabel("Time Steps")
        plt.ylabel("Amplitude")
        plt.show()
        fig.savefig('results/CAE_clean_scaled/reconstruction_delay.png', dpi=300)

    def search_max(self, loss_list, file_list):
        max_index = np.argmax(loss_list)
        print(max_index)
        max_name = file_list[max_index]
        print(f'最大値のファイル名: {max_name}')

    def write_csv(self, output_path, list):
        with open(output_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for i in list:
                writer.writerow([i])

    def weight(self, data, alpha):
        """
        重み付け関数
        """
        # パラメータ
        length = data.shape[-1]  # 長さ5000

        # 重みベクトルを作成
        x = torch.arange(length)
        weights = torch.exp(alpha * x)

        return data * weights

    def normalize(self, x):
        """
        正規化関数
        """
        # x: shape [B, 1, T]
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True)
        zscore = (x - mean) / std
        return zscore

if __name__ == '__main__':
    # データセットの準備
    csv_dir = ''  # CSVファイルのディレクトリ
    dataset = TimeSeriesDataset(csv_dir)

    # DataLoaderの設定
    batch_size = 1024 # バッチサイズを指定
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # DataLoaderの使用例
    for batch_idx, x_train in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}, Shape: {x_train.shape}")  # 出力: (B, 1, 20000)
