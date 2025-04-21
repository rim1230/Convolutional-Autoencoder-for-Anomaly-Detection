import os

# ファイルが存在するディレクトリを指定
directory = "./"

# ディレクトリ内のファイルを処理
for filename in os.listdir(directory):
    # 拡張子が .csv のファイルのみ対象
    if filename.endswith(".csv"):
        # ファイル名から拡張子を除去し、ゼロ埋めリネーム
        name, ext = os.path.splitext(filename)
        if name.isdigit():  # 数字だけのファイル名を対象
            new_name = f"{int(name):03d}{ext}"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            os.rename(old_path, new_path)

print("リネームが完了しました！")
