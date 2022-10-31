# エンコーディングに応じて適切に読み込むget_string_from_file関数
import chardet

def get_string_from_file(filename):
    with open(filename,'rb') as f:
        d = f.read()
        e = chardet.detect(d)['encoding']
        # 推定できなかったときはUTF-8で
        if e == None:
            e = 'UTF-8'
        return d.decode(e)