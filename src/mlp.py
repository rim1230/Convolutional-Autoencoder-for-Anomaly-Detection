import torch
import torch.nn as nn
import torch.optim as optimizers
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

class CAE(nn.Module):
    def __init__(self, input_dim, lr=1e-4, device=None):
        super(CAE, self).__init__()

        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lr = lr

        # エンコーダ
        self.encoder = nn.Sequential(
            nn.Linear(5000, 2500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2500, 1250),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1250, 625),
            nn.ReLU(),
        )

        # デコーダ
        self.decoder = nn.Sequential(
            nn.Linear(625, 1250),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1250, 2500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2500, 5000),
            nn.Dropout(0.2)
        )

        self.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optimizers.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def fit(self, train_loader, val_loader, epochs):
        # 学習ループ
        train_loss, val_loss = [], []
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
    
            # 学習フェーズ
            self.train()
            total_train_loss = 0.0
            for batch in tqdm(train_loader, desc="Training", leave=False):
                x_batch, _ = batch
                x_batch = x_batch.to(self.device)
                self.optimizer.zero_grad()

                reconstructed = self(x_batch)
                loss = self.criterion(reconstructed, x_batch)
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_loss.append(avg_train_loss)

            # 検証フェーズ
            self.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation", leave=False):
                    x_batch, _ = batch
                    x_batch = x_batch.to(self.device)
                    reconstructed = self(x_batch)
                    loss = self.criterion(reconstructed, x_batch)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_loss.append(avg_val_loss)

            print(f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        return train_loss, val_loss

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self(x.to(self.device)).cpu()

    def calculate_reconstruction_loss(self, data_loader):
        all_mae = []
        all_avg_mae = []

        df = pd.read_csv('~/tdr_ae/results/analysis/average.csv', header=None, nrows=5000)
        average = df.to_numpy().flatten()

        for batch in data_loader:
            x_batch, _ = batch
            x_batch_pred = self.predict(x_batch)
            x_batch = x_batch.cpu() # ホストメモリにコピー

            # average をバッチサイズに合わせて拡張
            batch_size, seq_len, feature_dim = x_batch.shape
            average_expanded = np.tile(average, (batch_size, seq_len, 1))  # バッチサイズ・シーケンス長に合わせる

            mae = np.mean(np.abs(x_batch_pred.detach().numpy() - x_batch.detach().numpy()), axis=(1, 2))
            avg_mae = np.mean(np.abs(average_expanded - x_batch.detach().numpy()), axis=(1, 2))
            all_mae.extend(mae) 
            all_avg_mae.extend(avg_mae) # リストの中の要素を展開して他のリストに追加する

        # # MAE損失のヒストグラムを描画
        # plt.hist(all_mae, bins=10000)
        # plt.yscale('log')
        # plt.xlabel("Test MAE loss")
        # plt.ylabel("Number of samples")
        # plt.title("Histogram of Test MAE Loss")
        # plt.show()

        # # MAE損失を縦軸、タイムスタンプを横軸としてプロット
        # start_time = "2024-01-17 16:48"  # 開始日時
        # end_time = "2024-01-31 02:59"    # 終了日時
        # # 1分おきのタイムスタンプを生成
        # timestamps = pd.date_range(start=start_time, end=end_time, freq="T")

        # plt.plot(timestamps, all_mae)
        # plt.xlabel('Timestamps')
        # plt.ylabel('MAE Loss')
        # plt.title("Test MAE Loss by Autoencoder")
        # plt.grid(True)
        # plt.xticks(rotation=45)  # 横軸のラベルを回転
        # plt.tight_layout()
        # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(15))  # 横軸ラベルを最大10個に制限
        # plt.show()

        # # MAE損失の最大値をしきい値 (threshold) とする
        # threshold = np.max(all_mae)
        # print("再構築エラーの閾値: ", threshold)

        # # MAE損失の最大値をしきい値 (threshold) とする
        # threshold = np.max(all_avg_mae)
        # print("再構築エラーの閾値: ", threshold)

        # # all_mae の中で最大値を持つインデックスを取得
        # max_index = np.argmax(all_mae)

        # print("最大値のインデックス: ", max_index)

        return all_mae, all_avg_mae

    def calculate_reconstrcution_loss_average(self, data_loader):
        df = pd.read_csv('~/tdr_ae/results/analysis/average_min.csv')
        average = df.to_numpy().flatten()
        average = average[63:]

        all_mae = []
        for batch in data_loader:
            x_batch, _ = batch
            x_batch_pred = self.predict(x_batch)
            # tensor[1024, 1, 936]
    
            # average をバッチサイズに合わせて拡張
            batch_size, seq_len, feature_dim = x_batch.shape
            average_expanded = np.tile(average, (batch_size, seq_len, 1))  # バッチサイズ・シーケンス長に合わせる
    
            # データをホストメモリにコピー
            x_batch = x_batch.cpu().detach().numpy()
            x_batch_pred = x_batch_pred.detach().numpy()
    
            # MAE の計算
            mae = np.mean(np.abs(average_expanded - x_batch), axis=(1, 2))
            all_mae.extend(mae)  # リストに追加

        # # MAE損失のヒストグラムを描画
        # plt.hist(all_mae, bins=10000)
        # plt.yscale('log')
        # plt.xlabel("Test MAE loss")
        # plt.ylabel("Number of samples")
        # plt.title("Histogram of Test MAE Loss")
        # plt.show()

        # # MAE損失を縦軸、タイムスタンプを横軸としてプロット
        # start_time = "2024-01-17 16:48"  # 開始日時
        # end_time = "2024-01-31 02:59"    # 終了日時
        # # 1分おきのタイムスタンプを生成
        # timestamps = pd.date_range(start=start_time, end=end_time, freq="T")

        # plt.plot(timestamps, all_mae)
        # plt.xlabel('Timestamps')
        # plt.ylabel('MAE Loss')
        # plt.title("Test MAE Loss by Average")
        # plt.grid(True)
        # plt.xticks(rotation=45)  # 横軸のラベルを回転
        # plt.tight_layout()
        # plt.gca().xaxis.set_major_locator(plt.MaxNLocator(15))  # 横軸ラベルを最大10個に制限
        # plt.show()

        # MAE損失の最大値をしきい値 (threshold) とする
        threshold = np.max(all_mae)
        print("再構築エラーの閾値: ", threshold)

        # # all_mae の中で最大値を持つインデックスを取得
        # max_index = np.argmax(all_mae)

        # print("最大値のインデックス: ", max_index)

        return all_mae


if __name__ == "__main__":
    # 使用例
    # データ準備
    x_train = torch.randn(312, 1, 936)  # ダミーデータ（312サンプル, チャネル数1, 長さ936）

    # モデル初期化と学習
    model = CAE(input_dim=936)
    print(model.device)
    train_loss, val_loss = model.fit(x_train, epochs=20)

    # 学習曲線のプロット
    model.plot_loss(train_loss, val_loss)

    # 再構成結果のプロット
    x_sample = x_train[0:1]
    reconstructed = model.predict(x_sample)
    model.plot_reconstruction(x_sample[0], reconstructed[0])
