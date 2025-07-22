import os
import re
from gensim.models.fasttext import load_facebook_vectors
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
import MeCab
from config import FASTTEXT_MODEL_PATH
import time

# ──────────────────────────────────────────────
# 1) MeCab トークナイザ（内容語だけ残す）
# ──────────────────────────────────────────────
ALLOWED_POS   = {'名詞', '動詞', '形容詞'}
NG_POS        = {'助詞', '助動詞', '接続詞', '連体詞', '副詞'}
NG_SUBPOS     = {'非自立', '副詞可能', '接尾'}

try:
    tagger = MeCab.Tagger('-Ochasen')
    def mecab_tokens(text: str):
        tokens = []
        for line in tagger.parse(text).splitlines()[:-2]:
            if '\t' in line:
                surface, _, _, pos, subpos, *_ = line.split('\t')
                if pos in NG_POS or pos not in ALLOWED_POS:
                    continue
                if subpos in NG_SUBPOS:
                    continue
                if len(surface) < 2:
                    continue
                tokens.append(surface)
        return tokens
except:
    def mecab_tokens(text: str):
        import re
        words = re.findall(r'[\u4e00-\u9fff]+', text)
        return [word for word in words if len(word) > 1]

vectorizer = TfidfVectorizer(
    tokenizer=mecab_tokens,
    token_pattern=None,
    ngram_range=(1, 2),
    min_df=1,
    max_df=1.0,
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=True
)

STOPWORDS = set([
    '発明', '装置', '方法', '請求', '本', '例', '図', '部', '手段', 'こと', 'もの', 'ため', 'よう', '及び',
    'により', 'において', 'に対し', 'に関する', 'に基づく', 'に先立ち', 'に伴い', 'に代えて', 'に加え', 'に応じ', 'に従い', 'に沿って', 'に比べ', 'に基づき',
    'また', 'さらに', 'この', 'その', 'これ', 'それ', 'および', 'ならびに', '等', '等々', '一方', '他方', '場合', '時', '際', '上', '下', '前', '後', '内', '外', '左', '右',
    '上記', '当該', '本発明', '該', '前記', 'この', 'その', 'これ', 'それ', '第'
])
COMMON_VERB_STOP = set([
    'する', '行う', 'なる', '設ける', '含む', '備える', '保持する', '可能にする'
])

COMMON_STOP = {
    'こと', 'よう', 'とも', 'ともに', 'および', 'この', 'その',
    'ため', '場合', '以上', 'また', '一方', 'さらに',
    '本発明', '目的', '手段', '効果', '図', '例', '実施', '部分',
    '選択図', '解決手段', '上記', '下記', '課題', '決定'
}
CUSTOM_STOP = set()
KEY_STOP = COMMON_STOP | CUSTOM_STOP

STOPWORDS_PREFIX = [
    '本発明', '上記', '当該', '前記', '該', 'この', 'その', 'これ', 'それ', '第'
]

# --- 高速化: fastTextモデルとKeyBERTをグローバルで1回だけロード ---
_ft_model = None
_kw_model = None

def get_fasttext_model():
    global _ft_model
    if _ft_model is None:
        if not os.path.exists(FASTTEXT_MODEL_PATH):
            raise FileNotFoundError(
                f"fastText 日本語モデル({FASTTEXT_MODEL_PATH}) が見つかりません。\n"
                "https://fasttext.cc/docs/en/crawl-vectors.html から "
                "'cc.ja.300.bin' をダウンロードし、配置してください。"
            )
        _ft_model = load_facebook_vectors(FASTTEXT_MODEL_PATH)
    return _ft_model

def get_keybert_model():
    global _kw_model
    if _kw_model is None:
        _kw_model = KeyBERT(model=get_fasttext_model())
    return _kw_model

CONNECTIVES = {'及び', 'または', 'および', 'ならびに'}

def clean_keyword(kw):
    # 先頭のストップワードを長い順で除去
    for sw in sorted(STOPWORDS_PREFIX, key=len, reverse=True):
        if kw.startswith(sw):
            kw = kw[len(sw):]
    # 汎用動詞のみのキーワードは除外
    if kw in COMMON_VERB_STOP:
        return ""
    # 接続詞単体やストップワード＋接続詞だけの語は除外
    if kw in CONNECTIVES or any(kw == sw + c for sw in STOPWORDS_PREFIX for c in CONNECTIVES):
        return ""
    return kw

def extract_noun_wo_verb_phrases(text):
    phrases = []
    try:
        node = tagger.parseToNode(text)
        while node:
            # 名詞
            if node.feature.startswith('名詞'):
                noun = node.surface
                next1 = node.next
                if next1 and next1.surface == 'を':
                    next2 = next1.next
                    if next2 and next2.feature.startswith('動詞'):
                        verb = next2.surface
                        phrase = f"{noun}を{verb}"
                        # ストップワード・クリーンアップ適用
                        phrase_clean = clean_keyword(phrase)
                        if phrase_clean and phrase_clean not in STOPWORDS:
                            phrases.append(phrase_clean)
            node = node.next
    except Exception:
        pass
    return phrases

def extract_keywords_keybert_fasttext(text: str, topn: int = 10, ft_path: str = FASTTEXT_MODEL_PATH):
    kw_model = get_keybert_model()
    candidates = kw_model.extract_keywords(
        text,
        vectorizer=vectorizer,
        keyphrase_ngram_range=(1, 2),
        nr_candidates=50,
        use_mmr=True,
        diversity=0.7,
        top_n=topn * 4
    )
    filtered = []
    for phrase, _score in candidates:
        w = phrase.strip()
        # 純ひらがな語
        if re.fullmatch(r'[ぁ-ゖ]+', w):
            continue
        # 長さ
        if not (2 <= len(w) <= 20):
            continue
        # 数字・記号
        if re.search(r'\d', w) or re.search(r'[\W_]', w):
            continue
        # ストップワード
        if w in KEY_STOP:
            continue
        # クリーンアップ
        w_clean = clean_keyword(w)
        if not w_clean or w_clean in STOPWORDS:
            continue
        filtered.append(w_clean)
        if len(filtered) >= topn:
            break
    # 名詞＋を＋動詞パターンも抽出
    noun_wo_verb_phrases = extract_noun_wo_verb_phrases(text)
    # 重複を除いて合成
    all_keywords = list(dict.fromkeys(filtered + noun_wo_verb_phrases))
    return all_keywords 

def extract_topics(data):
    # IPC分類のcode（例: A61B8/00）をtopicsに
    ipc_codes = [ipc['code'] for ipc in data.get('metadata', {}).get('classification_ipc', [])]
    # keywordsから「装置」「方法」「システム」などで終わる語をtopics候補に
    keywords = data.get('metadata', {}).get('keywords', [])
    topic_like = [kw for kw in keywords if kw.endswith(('装置', '方法', 'システム', '回路', '処理', '技術', '構造', '用途'))]
    # IPCコード＋topic_like語から重複なく3件程度
    topics = list(dict.fromkeys(ipc_codes + topic_like))[:3]
    return topics 
