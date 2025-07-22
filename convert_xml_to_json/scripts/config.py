# config.py

# 入力ディレクトリ
INPUT_DIR = 'input_files/result_1/0'
# 出力ディレクトリ
OUTPUT_DIR = 'data/output_files'
# fastTextモデルパス
FASTTEXT_MODEL_PATH = 'bin/cc.ja.300.bin'
# バージョン
VERSION = 'v1.0.0'
# XML名前空間
NAMESPACES = {
    'jppat': 'http://www.jpo.go.jp/standards/XMLSchema/ST96/JPPatent',
    'com': 'http://www.wipo.int/standards/XMLSchema/ST96/Common',
    'pat': 'http://www.wipo.int/standards/XMLSchema/ST96/Patent',
    'jpcom': 'http://www.jpo.go.jp/standards/XMLSchema/ST96/JPCommon',
    'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
    'xsd': 'http://www.w3.org/2001/XMLSchema',
    'jp': 'http://www.jpo.go.jp', 
}
# 参照判定用CSVファイル
REFERENCE_CSV_FILES = ['data/himotsuki_csv/CSV1.csv', 'data/himotsuki_csv/CSV2.csv']
# キーワード抽出パラメータ
KEYWORD_TOPN = 10
KEYWORD_NGRAM_RANGE = (1, 2)
KEYWORD_MIN_DF = 1
KEYWORD_MAX_DF = 1.0 