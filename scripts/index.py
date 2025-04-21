import os
import csv
from tqdm import tqdm

def main():
    # ======== 設定部 ========
    # 親ディレクトリのパスを指定
    parent_dir = "./"
    # CSV出力先ファイル名
    output_csv = "/home/takayanagi/tdr_ae/src/index.csv"
    # ディレクトリ名のしきい値
    threshold_dir_name = "201901172359"
    # 除外するディレクトリ名
    exclude_dir_name = "DefectiveData"
    # ========================

    # 親ディレクトリ内のサブディレクトリを走査し、
    # 条件に合うディレクトリを抽出する
    target_dirs = []
    entries = os.listdir(parent_dir)
    entries.sort()
    for entry in entries:
        entry_path = os.path.join(parent_dir, entry)
        # ディレクトリであることを確認
        if os.path.isdir(entry_path):
            dir_name = entry
            # 条件1: dir_name > threshold_dir_name
            # 条件2: dir_name != exclude_dir_name
            # ここでは文字列比較でOK(タイムスタンプのような形式なら、単純比較で大小関係が保たれる)
            if dir_name > threshold_dir_name and dir_name != exclude_dir_name:
                target_dirs.append(entry_path)

    # ファイル情報(連番, ディレクトリ名, ファイル名)を格納するリスト
    file_info_list = []
    current_id = 0  # 通し番号の初期値
    # 抽出したディレクトリを順番に処理
    for d in tqdm(target_dirs, desc="Processing directories"):
        # ディレクトリ内のファイル一覧を取得(ファイル名の昇順でソート)
        files = sorted(os.listdir(d))
        for f in files:
            file_path = os.path.join(d, f)
            # ファイルかどうか確認(サブディレクトリはここでは無視)
            if os.path.isfile(file_path):
                file_info_list.append([current_id, os.path.basename(d), f])
                current_id += 1
    # # 抽出したディレクトリを順番に処理
    # for d in target_dirs:
    #     # ディレクトリ内のファイル一覧を取得(ファイル名の昇順でソート)
    #     files = sorted(os.listdir(d))
    #     for f in files:
    #         file_path = os.path.join(d, f)
    #         # ファイルかどうか確認(サブディレクトリはここでは無視)
    #         if os.path.isfile(file_path):
    #             file_info_list.append([current_id, os.path.basename(d), f])
    #             current_id += 1

    # CSVに書き出す
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # ヘッダーを書き込む (必要に応じてカスタマイズ)
        writer.writerow(["ID", "Directory", "Filename"])
        # データを書き込む
        writer.writerows(file_info_list)

    print(f"完了: {output_csv} に {len(file_info_list)} 件のレコードを書き込みました。")


if __name__ == "__main__":
    main()
