import csv
import torch
import numpy as np
import json
import os
from cae import CAE
from data import Data
from tqdm import tqdm
from torchinfo import summary
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = json.load(f)

# 出力ディレクトリの作成
output_dir = config["output"]["dir"]
if os.path.exists(output_dir):
    print(f"Output directory {output_dir} already exists. Stopping the program.")
    exit()
else:
    os.makedirs(output_dir)

# JSONデータを別のファイルに保存
with open(os.path.join(config["output"]["dir"], "../configs/sample.json"), "w") as f:
    json.dump(config, f, indent=2)

# 再現性確保のためのシード値
seed = config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# インスタンス化
data = Data(batch_size=config["data"]["batch_size"], test_batch_size=config["data"]["test_batch_size"], val_split=config["data"]["val_split"]) # 1440 ... 1日分のデータ数

# 1/17 23:00-23:59 のデータを訓練用にロードする
directory = config["data"]["data_dir"]
file_list = data.make_file_list(directory)
x_train = data.next_batch(directory, file_list)

# データローダーの作成
train_loader, val_loader = data.prepare_dataset(x_train)

# 訓練データの確認
for x, y in train_loader:
    print(x.shape, y.shape)

# モデル初期化と学習
model = CAE(input_dim=config["model"]["input_dim"])
summary(model, input_size=(data.batch_size, 1, config["model"]["input_dim"]))

print(model.device)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

train_loss, val_loss = model.fit(train_loader, val_loader, epochs=config["training"]["epochs"])

# 学習曲線のプロット
data.plot_loss(train_loss, val_loss)

directory = "" # テストデータのディレクトリ
file_list = data.make_file_list(directory)
x_test = data.load_data_from_directory(directory, file_list, column_index=3, skip_rows=1, nrows=5000)
test_loader = data.prepare_test_dataset(x_test, batch_size=312)
model.predict_save(test_loader, "1hour")

# テストデータの再構成誤差を計算
test_mae = model.calculate_reconstruction_loss(test_loader)
output_filename = os.path.join(config["output"]["dir"], "recon_loss_test.csv")
data.write_csv(output_filename, test_mae)
