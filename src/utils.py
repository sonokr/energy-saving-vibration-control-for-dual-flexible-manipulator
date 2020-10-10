import os


def create_dirs(dirpath):
    """ディレクトリが存在しなければ作成し，存在すれば何もしない
    """
    os.makedirs(dirpath, exist_ok=True)
