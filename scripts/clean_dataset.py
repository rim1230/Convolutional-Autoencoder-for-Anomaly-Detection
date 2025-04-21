import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import shutil
import os
from tqdm import tqdm



dfs = []
for day in range(18, 31):
    df_day = pd.read_csv(f'/home/takayanagi/tdr_ae/results/CAE_scaled/merged/01{day}.csv', header=None)
    dfs.append(df_day)
df = pd.concat(dfs, ignore_index=True)
df.columns = ['err']

# 閾値の設定
threshold = 0.035

# 異常なデータのIDを取得
anomaly_id = df[df['err'] > threshold].index



# index.csv の読み込み
index_df = pd.read_csv('/home/takayanagi/tdr_ae/scripts/index.csv')
index_df = index_df.iloc[432:19152]
# index.csvは1/17からはじまっているので、1/18からのIDに変換
index_df['end_id'] = index_df['end_id'] - 154846

# IDの整合をとる
index_df['end_id'] = index_df['end_id'] - 1

# start_id 計算
# 1行目の start_id は1、2行目以降は (前の行のend_id+1)
index_df['start_id'] = index_df['end_id'].shift(1, fill_value=-1) + 1


# rawデータをmvする
# 異常データのIDから、対応するファイルパスを取得し，moveする
for target_id in tqdm(anomaly_id, desc="Processing anomalies"):
    # 移動するファイルのパスを取得
    row = index_df[index_df['end_id'] >= target_id].iloc[0]
    start_id = row['start_id']
    offset = target_id - start_id  # CSV内の行オフセット(0始まり)
    dir = row['dir_path']

    # ディレクトリ内のoffset番目のファイルを削除
    files = sorted(os.listdir(dir))
    file_to_move = os.path.join(dir, files[offset])

    # ファイルをDefectiveDataディレクトリに移動
    shutil.move(file_to_move, os.path.join('DefectiveData', f'{files[offset]}.csv'))
    print(f'moved {file_to_move} to DefectiveData/{offset}.csv')

# # diffデータで異常データを削除したCSVを作成
# for target_id in tqdm(anomaly_id[10:], desc="Processing anomalies"):
#     # 移動するファイルのパスを取得
#     row = index_df[index_df['end_id'] >= target_id].iloc[0]
#     start_id = row['start_id']
#     offset = target_id - start_id  # CSV内の行オフセット(0始まり)
#     dir = row['dir_path']

#     df = pd.read_csv(f'/home/takayanagi/data/diff/kyoto/all/{dir}.csv', header=None)
#     df = df.drop(offset)
#     df.to_csv(f'/home/takayanagi/data/diff/kyoto/all/{dir}.csv', header=None, index=None)