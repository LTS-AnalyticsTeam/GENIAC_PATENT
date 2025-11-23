#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1. create_graph.py - データ準備段階（構造化版）

特許JSONファイルからNeo4jグラフデータベースを構築
セクションの階層構造を保持して格納（tech-problem, tech-solution, advantageous-effectsを個別ノード化）
埋め込みベクトルを生成してベクトルインデックスを作成
トピック（キーワード）ノードを生成
"""

import os
import sys
import json
import argparse
import time
import hashlib
import re
import threading
from typing import List, Dict, Set, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase
from tqdm import tqdm

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("警告: tiktoken が利用できません。pip install tiktoken を実行してください。")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("警告: numpy が利用できません。pip install numpy を実行してください。")

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASS = os.environ["NEO4J_PASS"]
EMB_MODEL = "text-embedding-3-small"

# 最適化設定（長文対応）
BATCH_SIZE = 500
EMBEDDING_BATCH_SIZE = 30
MAX_WORKERS = 4
MAX_TOKENS = 7500
CHUNK_OVERLAP = 100

DEFAULT_STOPWORDS = {
    "もの", "こと", "ため", "とき", "場合", "方法", "手段", "装置", "システム",
    "部分", "全体", "一部", "他方", "一方", "双方", "各々", "それぞれ",
    "状態", "情況", "条件", "環境", "状況", "例", "実例", "具体例",
    "動作", "操作", "処理", "機能", "性能", "効果", "結果", "影響",
    "特徴", "性質", "属性", "要素", "成分", "構成", "構造", "形状",
    "発明", "実施", "実施例", "形態", "態様", "変形", "変形例", "応用",
    "利用", "使用", "適用", "採用", "選択", "決定", "判断", "制御"
}

@dataclass
class PatentData:
    pub_id: str
    title: str
    ipc_codes: List[str]
    sections: Dict[str, str]

@dataclass 
class SectionData:
    node_id: str
    section_type: str
    pub_id: str
    text: str
    parent_section: str = None  # 親セクション情報を追加
    embedding: List[float] = None
    is_chunked: bool = False
    chunk_count: int = 1

@dataclass
class TopicData:
    topic_id: str
    text: str
    section_type: str
    pub_id: str
    section_node_id: str
    embedding: List[float] = None

class EnhancedLongTextGraphCreator:
    """構造化セクション対応グラフ作成器"""
    
    def __init__(self, stopwords_file: str = None, max_workers: int = MAX_WORKERS, incremental: bool = True):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USER, NEO4J_PASS),
            max_connection_lifetime=300,
            max_connection_pool_size=50
        )
        self.stopwords = self._load_stopwords(stopwords_file)
        self.max_workers = max_workers
        self.incremental = incremental
        
        # 既存データ管理
        self.existing_patents = set()
        self.processed_files_cache = set()
        
        # 統計情報
        self.stats = {
            'total_files': 0,
            'skipped_files': 0,
            'processed_files': 0,
            'created_patents': 0,
            'created_sections': 0,
            'created_topics': 0,
            'filtered_stopwords': 0,
            'embedding_calls': 0,
            'chunked_texts': 0,
            'total_chunks': 0,
            'structured_sections': 0  # 構造化セクション数
        }
        
        # スレッドロック
        self.stats_lock = threading.Lock()
        self.embedding_lock = threading.Lock()
        
        # トークンエンコーダー初期化
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except:
                self.encoding = None
        else:
            self.encoding = None
        
        # 増分処理の場合は既存データを取得
        if self.incremental:
            self._load_existing_data()
        
        mode = "増分処理" if incremental else "完全再作成"
        print(f"{mode}モード設定: workers={max_workers}, batch_size={BATCH_SIZE}")
        print(f"構造化セクション対応: tech-problem, tech-solution, advantageous-effects")
        print(f"最大トークン数: {MAX_TOKENS}, ストップワード: {len(self.stopwords)}個")
        print(f"tiktoken: {'利用可能' if self.encoding else '利用不可'}, numpy: {'利用可能' if NUMPY_AVAILABLE else '利用不可'}")
        
        if incremental:
            print(f"既存特許データ: {len(self.existing_patents)}件")
    
    def _load_existing_data(self):
        """既存データ読み込み（増分処理用）"""
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (p:Patent) RETURN p.pub_id as pub_id")
                self.existing_patents = {record["pub_id"] for record in result}
                print(f"既存特許データを読み込みました: {len(self.existing_patents)}件")
        except Exception as e:
            print(f"既存データ読み込みエラー: {e}")
            print("新規作成モードで継続します")
            self.existing_patents = set()
    
    def _extract_pub_id_from_file(self, filepath: str) -> str:
        """ファイルから特許ID抽出"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            bibl = data.get("bibliographic", {})
            pub_id = bibl.get("publication", {}).get("doc_number", "")
            return pub_id
        except Exception:
            return ""
    
    def _should_process_file(self, filepath: Path) -> Tuple[bool, str]:
        """ファイル処理要否判定"""
        if not self.incremental:
            return True, "full_rebuild"
        
        file_path_str = str(filepath)
        
        if file_path_str in self.processed_files_cache:
            return False, "cached_skip"
        
        pub_id = self._extract_pub_id_from_file(file_path_str)
        
        if not pub_id:
            print(f"警告: 特許ID抽出失敗 {filepath.name}")
            return False, "invalid_file"
        
        if pub_id in self.existing_patents:
            self.processed_files_cache.add(file_path_str)
            return False, "already_exists"
        
        return True, "new_patent"
    
    def _load_stopwords(self, stopwords_file: str = None) -> Set[str]:
        """ストップワード読み込み"""
        stopwords = DEFAULT_STOPWORDS.copy()
        
        if stopwords_file and Path(stopwords_file).exists():
            try:
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    file_stopwords = set(line.strip() for line in f if line.strip())
                stopwords.update(file_stopwords)
                print(f"ストップワードファイル読み込み: +{len(file_stopwords)}個")
            except Exception as e:
                print(f"ストップワードファイル読み込みエラー: {e}")
        
        return stopwords
    
    def is_stopword(self, word: str) -> bool:
        """ストップワード判定"""
        if not word or len(word.strip()) <= 1:
            return True
        
        word_clean = word.strip()
        
        if word_clean in self.stopwords:
            return True
        
        if (re.match(r'^[0-9]+$', word_clean) or 
            re.match(r'^[^\w\s]+$', word_clean) or
            re.match(r'^[ぁ-ん]{1,2}$', word_clean) or
            re.match(r'^[a-zA-Z]$', word_clean)):
            return True
        
        return False
    
    def count_tokens(self, text: str) -> int:
        """テキストのトークン数をカウント"""
        if self.encoding:
            try:
                return len(self.encoding.encode(text))
            except:
                pass
        
        japanese_chars = len(re.findall(r'[ぁ-んァ-ン一-龯]', text))
        other_chars = len(text) - japanese_chars
        return int(japanese_chars * 1.5 + other_chars * 0.25)
    
    def smart_text_splitter(self, text: str, max_tokens: int = MAX_TOKENS) -> List[str]:
        """スマートテキスト分割（オーバーラップ付き）"""
        if self.count_tokens(text) <= max_tokens:
            return [text]
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if self.count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                if self.count_tokens(paragraph) > max_tokens:
                    sentences = self._split_by_sentences(paragraph, max_tokens)
                    chunks.extend(sentences[:-1])
                    current_chunk = sentences[-1] if sentences else ""
                else:
                    current_chunk = paragraph
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        if len(chunks) > 1:
            chunks = self._add_overlap(chunks)
        
        final_chunks = []
        for chunk in chunks:
            if self.count_tokens(chunk) <= max_tokens:
                final_chunks.append(chunk)
            else:
                final_chunks.append(self._force_truncate(chunk, max_tokens))
        
        return final_chunks
    
    def _split_by_sentences(self, text: str, max_tokens: int) -> List[str]:
        """文単位での分割"""
        sentences = re.split(r'[。．！？]\s*', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            test_chunk = current_chunk + sentence + "。" if current_chunk else sentence + "。"
            
            if self.count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                if self.count_tokens(sentence + "。") > max_tokens:
                    current_chunk = self._force_truncate(sentence + "。", max_tokens)
                else:
                    current_chunk = sentence + "。"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """チャンク間のオーバーラップを追加"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            prev_sentences = re.split(r'[。．]\s*', prev_chunk)
            if len(prev_sentences) > 1:
                overlap_text = prev_sentences[-2] + "。" if len(prev_sentences) >= 2 else ""
                
                if overlap_text and self.count_tokens(overlap_text + current_chunk) <= MAX_TOKENS:
                    overlapped_chunk = overlap_text + "\n" + current_chunk
                    overlapped_chunks.append(overlapped_chunk)
                else:
                    overlapped_chunks.append(current_chunk)
            else:
                overlapped_chunks.append(current_chunk)
        
        return overlapped_chunks
    
    def _force_truncate(self, text: str, max_tokens: int) -> str:
        """強制切り詰め（最後の手段）"""
        while self.count_tokens(text) > max_tokens and len(text) > 50:
            cut_point = int(len(text) * 0.9)
            while cut_point > 0 and text[cut_point] not in '。．\n ':
                cut_point -= 1
            
            if cut_point <= 0:
                cut_point = int(len(text) * 0.9)
            
            text = text[:cut_point]
        
        return text
    
    def embed_text_with_chunking(self, text: str) -> Tuple[List[float], bool, int]:
        """チャンク分割対応埋め込み生成"""
        chunks = self.smart_text_splitter(text)
        
        if len(chunks) == 1:
            try:
                response = self.client.embeddings.create(model=EMB_MODEL, input=[chunks[0]])
                return response.data[0].embedding, False, 1
            except Exception as e:
                print(f"埋め込み生成エラー: {e}")
                return [], False, 0
        
        print(f"長文を{len(chunks)}チャンクに分割処理")
        
        chunk_embeddings = []
        for i, chunk in enumerate(chunks):
            try:
                response = self.client.embeddings.create(model=EMB_MODEL, input=[chunk])
                chunk_embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"チャンク{i+1}埋め込み生成エラー: {e}")
                continue
        
        if not chunk_embeddings:
            return [], True, len(chunks)
        
        if NUMPY_AVAILABLE:
            avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
        else:
            embedding_dim = len(chunk_embeddings[0])
            avg_embedding = []
            for dim in range(embedding_dim):
                avg_value = sum(emb[dim] for emb in chunk_embeddings) / len(chunk_embeddings)
                avg_embedding.append(avg_value)
        
        with self.stats_lock:
            self.stats['chunked_texts'] += 1
            self.stats['total_chunks'] += len(chunks)
        
        print(f"チャンク埋め込み平均化完了: {len(chunk_embeddings)}/{len(chunks)}個のベクトル")
        return avg_embedding, True, len(chunks)
    
    def extract_keywords_optimized(self, text: str, max_keywords: int = 20) -> List[str]:
        """最適化キーワード抽出"""
        patterns = [
            r'[ァ-ヴー]{2,}',
            r'[一-龯]{2,}',
            r'[A-Za-z0-9]{2,}',
            r'[ぁ-んァ-ヴー一-龯]{3,}'
        ]
        
        all_candidates = []
        for pattern in patterns:
            all_candidates.extend(re.findall(pattern, text))
        
        term_freq = Counter()
        for term in all_candidates:
            if not self.is_stopword(term) and 2 <= len(term) <= 20:
                term_freq[term] += 1
        
        return [term for term, freq in term_freq.most_common(max_keywords)]
    
    def extract_text_from_section(self, section_obj) -> str:
        """セクションテキスト抽出"""
        texts = []
        
        def walk(obj):
            if isinstance(obj, dict):
                if "text" in obj and obj["text"]:
                    texts.append(str(obj["text"]).strip())
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)
        
        walk(section_obj)
        return "\n".join(texts)
    
    def parse_patent_file(self, filepath: str) -> PatentData:
        """特許ファイル解析（構造化対応版）"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"ファイル読み込みエラー {filepath}: {e}")
            return None
        
        bibl = data.get("bibliographic", {})
        pub_id = bibl.get("publication", {}).get("doc_number", "")
        title = bibl.get("title", "")
        ipc_data = bibl.get("classification", {}).get("ipc", [])
        ipc_codes = [ipc.get("text", "") for ipc in ipc_data]
        
        if not pub_id:
            return None
        
        desc = data.get("description", {})
        sections = {}
        
        # 基本セクション（technical-field, background-art）
        for section_type in ["technical-field", "background-art"]:
            if section_type in desc:
                text = self.extract_text_from_section(desc[section_type])
                if text:
                    sections[section_type] = text
        
        # summary-of-invention の子要素を個別に抽出（構造化）
        summary = desc.get("summary-of-invention", [])
        
        # リスト形式の場合は最初の要素を取得
        if isinstance(summary, list) and summary:
            summary = summary[0]
        
        if isinstance(summary, dict):
            # 子要素（tech-problem, tech-solution, advantageous-effects）を個別抽出
            for child_section in ["tech-problem", "tech-solution", "advantageous-effects"]:
                if child_section in summary:
                    text = self.extract_text_from_section(summary[child_section])
                    if text:
                        sections[child_section] = text
                        with self.stats_lock:
                            self.stats['structured_sections'] += 1
        
        return PatentData(pub_id, title, ipc_codes, sections)
    
    def process_patent_data(self, patent_data: PatentData) -> Tuple[List[SectionData], List[TopicData]]:
        """特許データ処理（長文対応）"""
        sections = []
        topics = []
        
        # 親セクション判定
        structured_child_sections = {"tech-problem", "tech-solution", "advantageous-effects"}
        
        # セクション処理
        for section_type, text in patent_data.sections.items():
            node_id = hashlib.md5(f"{patent_data.pub_id}:{section_type}:{text[:100]}".encode()).hexdigest()
            
            # 親セクション情報
            parent_section = "summary-of-invention" if section_type in structured_child_sections else None
            
            # 長文対応埋め込み生成
            embedding, is_chunked, chunk_count = self.embed_text_with_chunking(text)
            
            section_obj = SectionData(
                node_id=node_id,
                section_type=section_type,
                pub_id=patent_data.pub_id,
                text=text,
                parent_section=parent_section,
                embedding=embedding,
                is_chunked=is_chunked,
                chunk_count=chunk_count
            )
            
            sections.append(section_obj)
            
            # トピック処理
            if embedding:
                keywords = self.extract_keywords_optimized(text)
                
                with self.stats_lock:
                    original_count = len(re.findall(r'[ぁ-んァ-ヴー一-龯A-Za-z0-9]{2,}', text))
                    self.stats['filtered_stopwords'] += original_count - len(keywords)
                
                for keyword in keywords:
                    topic_id = f"{patent_data.pub_id}_{section_type}_{hashlib.md5(keyword.encode()).hexdigest()[:8]}"
                    
                    try:
                        response = self.client.embeddings.create(model=EMB_MODEL, input=[keyword])
                        keyword_embedding = response.data[0].embedding
                    except:
                        keyword_embedding = []
                    
                    if keyword_embedding:
                        topic_obj = TopicData(
                            topic_id=topic_id,
                            text=keyword,
                            section_type=section_type,
                            pub_id=patent_data.pub_id,
                            section_node_id=node_id,
                            embedding=keyword_embedding
                        )
                        topics.append(topic_obj)
        
        return sections, topics
    
    def clear_existing_data(self):
        """既存データ完全削除（再作成モード用）"""
        print("既存データを削除中...")
        
        with self.driver.session() as session:
            try:
                session.run("MATCH (n) DETACH DELETE n")
                print("✓ 既存データ削除完了")
                
                session.run("DROP INDEX sectionEmbeddingIdx IF EXISTS")
                session.run("DROP INDEX topicEmbeddingIdx IF EXISTS")
                session.run("DROP INDEX patent_pub_id IF EXISTS")
                session.run("DROP INDEX section_type IF EXISTS")
                session.run("DROP INDEX section_pub_id IF EXISTS")
                session.run("DROP INDEX topic_section_type IF EXISTS")
                print("✓ インデックス削除完了")
                
            except Exception as e:
                print(f"データ削除エラー: {e}")
    
    def create_patents_batch(self, tx, patent_data_list: List[PatentData]):
        """特許ノードバッチ作成"""
        batch_data = []
        for patent in patent_data_list:
            batch_data.append({
                'pub_id': patent.pub_id,
                'title': patent.title,
                'ipc_codes': patent.ipc_codes
            })
        
        tx.run("""
            UNWIND $batch AS row
            MERGE (p:Patent {pub_id: row.pub_id})
            SET p.title = row.title,
                p.ipc_codes = row.ipc_codes,
                p.updated_at = timestamp()
        """, batch=batch_data)
    
    def create_sections_batch(self, tx, sections: List[SectionData]):
        """セクションノードバッチ作成（parent_section対応）"""
        valid_sections = [s for s in sections if s.embedding]
        
        if not valid_sections:
            return
        
        batch_data = []
        for section in valid_sections:
            batch_data.append({
                'node_id': section.node_id,
                'section_type': section.section_type,
                'pub_id': section.pub_id,
                'text': section.text,
                'parent_section': section.parent_section,
                'embedding': section.embedding,
                'is_chunked': section.is_chunked,
                'chunk_count': section.chunk_count
            })
        
        tx.run("""
            UNWIND $batch AS row
            MERGE (s:Section {node_id: row.node_id})
            SET s.section_type = row.section_type,
                s.pub_id = row.pub_id,
                s.text = row.text,
                s.parent_section = row.parent_section,
                s.embedding = row.embedding,
                s.is_chunked = row.is_chunked,
                s.chunk_count = row.chunk_count,
                s.updated_at = timestamp()
        """, batch=batch_data)
        
        tx.run("""
            UNWIND $batch AS row
            MATCH (p:Patent {pub_id: row.pub_id})
            MATCH (s:Section {node_id: row.node_id})
            MERGE (p)-[:HAS_SECTION {type: row.section_type}]->(s)
        """, batch=batch_data)
    
    def create_topics_batch(self, tx, topics: List[TopicData]):
        """トピックノードバッチ作成"""
        valid_topics = [t for t in topics if t.embedding]
        
        if not valid_topics:
            return
        
        batch_data = []
        for topic in valid_topics:
            batch_data.append({
                'topic_id': topic.topic_id,
                'text': topic.text,
                'section_type': topic.section_type,
                'pub_id': topic.pub_id,
                'section_node_id': topic.section_node_id,
                'embedding': topic.embedding
            })
        
        tx.run("""
            UNWIND $batch AS row
            MERGE (t:Topic {topic_id: row.topic_id})
            SET t.text = row.text,
                t.section_type = row.section_type,
                t.pub_id = row.pub_id,
                t.embedding = row.embedding,
                t.updated_at = timestamp()
        """, batch=batch_data)
        
        tx.run("""
            UNWIND $batch AS row
            MATCH (s:Section {node_id: row.section_node_id})
            MATCH (t:Topic {topic_id: row.topic_id})
            MERGE (s)-[:HAS_TOPIC]->(t)
        """, batch=batch_data)
    
    def create_section_relationships_batch(self, tx, patent_ids: List[str]):
        """セクション間関係バッチ作成"""
        tx.run("""
            UNWIND $patent_ids AS pub_id
            MATCH (p:Patent {pub_id: pub_id})-[:HAS_SECTION]->(tf:Section {section_type: 'technical-field'})
            MATCH (p)-[:HAS_SECTION]->(ba:Section {section_type: 'background-art'})
            MERGE (tf)-[:PROVIDES_CONTEXT_FOR]->(ba)
        """, patent_ids=patent_ids)
        
        # 構造化セクション間の関係
        tx.run("""
            UNWIND $patent_ids AS pub_id
            MATCH (p:Patent {pub_id: pub_id})-[:HAS_SECTION]->(prob:Section {section_type: 'tech-problem'})
            MATCH (p)-[:HAS_SECTION]->(sol:Section {section_type: 'tech-solution'})
            MERGE (prob)-[:SOLVED_BY]->(sol)
        """, patent_ids=patent_ids)
        
        tx.run("""
            UNWIND $patent_ids AS pub_id
            MATCH (p:Patent {pub_id: pub_id})-[:HAS_SECTION]->(sol:Section {section_type: 'tech-solution'})
            MATCH (p)-[:HAS_SECTION]->(eff:Section {section_type: 'advantageous-effects'})
            MERGE (sol)-[:PRODUCES]->(eff)
        """, patent_ids=patent_ids)
    
    def create_indexes(self):
        """インデックス作成"""
        with self.driver.session() as session:
            print("基本インデックス作成中...")
            
            session.run("CREATE INDEX patent_pub_id IF NOT EXISTS FOR (p:Patent) ON (p.pub_id)")
            session.run("CREATE INDEX section_type IF NOT EXISTS FOR (s:Section) ON (s.section_type)")
            session.run("CREATE INDEX section_pub_id IF NOT EXISTS FOR (s:Section) ON (s.pub_id)")
            session.run("CREATE INDEX topic_section_type IF NOT EXISTS FOR (t:Topic) ON (t.section_type)")
            
            print("ベクトルインデックス作成中...")
            start_time = time.time()
            
            try:
                session.run("""
                    CREATE VECTOR INDEX sectionEmbeddingIdx IF NOT EXISTS
                    FOR (s:Section) ON (s.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
                
                session.run("""
                    CREATE VECTOR INDEX topicEmbeddingIdx IF NOT EXISTS
                    FOR (t:Topic) ON (t.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }}
                """)
                
                elapsed = time.time() - start_time
                print(f"ベクトルインデックス作成完了: {elapsed:.2f}秒")
                
            except Exception as e:
                print(f"ベクトルインデックス作成エラー: {e}")
    
    def validate_data_quality(self, all_patents: List[PatentData], all_sections: List[SectionData], all_topics: List[TopicData]):
        """データ品質チェック"""
        print("データ品質チェック実行中...")
        
        section_count = defaultdict(int)
        patents_with_sections = defaultdict(set)
        chunked_sections = 0
        total_chunks = 0
        
        for section in all_sections:
            if section.embedding:
                section_count[section.section_type] += 1
                patents_with_sections[section.pub_id].add(section.section_type)
                
                if section.is_chunked:
                    chunked_sections += 1
                    total_chunks += section.chunk_count
        
        print(f"セクション分布:")
        for section_type, count in section_count.items():
            print(f"  {section_type}: {count}件")
        
        complete_patents = sum(1 for sections in patents_with_sections.values() if len(sections) >= 2)
        
        print(f"完全なセクションを持つ特許: {complete_patents}/{len(all_patents)}")
        print(f"長文分割処理: {chunked_sections}件 (平均{total_chunks/max(chunked_sections,1):.1f}チャンク)")
        
        sections_with_embeddings = len([s for s in all_sections if s.embedding])
        topics_with_embeddings = len([t for t in all_topics if t.embedding])
        
        print(f"埋め込み生成成功率:")
        print(f"  セクション: {sections_with_embeddings}/{len(all_sections)} ({sections_with_embeddings/len(all_sections)*100:.1f}%)")
        print(f"  トピック: {topics_with_embeddings}/{len(all_topics)} ({topics_with_embeddings/len(all_topics)*100:.1f}%)")
        
        return complete_patents > 0
    
    def process_files_smart(self, files: List[Path], limit: int = 0):
        """スマートファイル処理（増分対応）"""
        if limit and len(files) > limit:
            files = files[:limit]
        
        self.stats['total_files'] = len(files)
        print(f"対象ファイル: {len(files)}件")
        
        if not self.incremental:
            self.clear_existing_data()
        
        files_to_process = []
        skip_reasons = defaultdict(int)
        
        print("ファイル処理要否判定中...")
        for filepath in tqdm(files, desc="判定"):
            should_process, reason = self._should_process_file(filepath)
            
            if should_process:
                files_to_process.append(filepath)
            else:
                skip_reasons[reason] += 1
                with self.stats_lock:
                    self.stats['skipped_files'] += 1
        
        if skip_reasons:
            print(f"スキップファイル統計:")
            for reason, count in skip_reasons.items():
                reason_msg = {
                    'cached_skip': '既処理(キャッシュ)',
                    'already_exists': '既存特許',
                    'invalid_file': '無効ファイル'
                }.get(reason, reason)
                print(f"  {reason_msg}: {count}件")
        
        print(f"処理対象ファイル: {len(files_to_process)}件 (スキップ: {self.stats['skipped_files']}件)")
        
        if not files_to_process:
            print("新規処理対象ファイルがありません")
            self.print_statistics()
            return
        
        print("段階1: データ解析と埋め込み生成...")
        
        all_patents = []
        all_sections = []
        all_topics = []
        
        def process_single_file(filepath):
            patent_data = self.parse_patent_file(str(filepath))
            if not patent_data:
                return None, [], []
            
            sections, topics = self.process_patent_data(patent_data)
            
            with self.stats_lock:
                self.stats['processed_files'] += 1
                self.stats['created_sections'] += len(sections)
                self.stats['created_topics'] += len(topics)
            
            return patent_data, sections, topics
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_single_file, f) for f in files_to_process]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="データ解析"):
                patent_data, sections, topics = future.result()
                if patent_data:
                    all_patents.append(patent_data)
                    all_sections.extend(sections)
                    all_topics.extend(topics)
        
        print(f"解析完了: 特許{len(all_patents)}件, セクション{len(all_sections)}件, トピック{len(all_topics)}件")
        
        if not self.validate_data_quality(all_patents, all_sections, all_topics):
            print("データ品質チェックに失敗しました。処理を中止します。")
            return
        
        print("段階2: データベース投入（バッチ処理）...")
        
        with self.driver.session() as session:
            for i in tqdm(range(0, len(all_patents), BATCH_SIZE), desc="特許作成"):
                batch = all_patents[i:i + BATCH_SIZE]
                try:
                    session.execute_write(self.create_patents_batch, batch)
                    with self.stats_lock:
                        self.stats['created_patents'] += len(batch)
                except Exception as e:
                    print(f"特許作成エラー (batch {i}): {e}")
            
            for i in tqdm(range(0, len(all_sections), BATCH_SIZE), desc="セクション作成"):
                batch = all_sections[i:i + BATCH_SIZE]
                try:
                    session.execute_write(self.create_sections_batch, batch)
                except Exception as e:
                    print(f"セクション作成エラー (batch {i}): {e}")
            
            for i in tqdm(range(0, len(all_topics), BATCH_SIZE), desc="トピック作成"):
                batch = all_topics[i:i + BATCH_SIZE]
                try:
                    session.execute_write(self.create_topics_batch, batch)
                except Exception as e:
                    print(f"トピック作成エラー (batch {i}): {e}")
            
            patent_ids = [p.pub_id for p in all_patents]
            for i in tqdm(range(0, len(patent_ids), BATCH_SIZE), desc="関係作成"):
                batch = patent_ids[i:i + BATCH_SIZE]
                try:
                    session.execute_write(self.create_section_relationships_batch, batch)
                except Exception as e:
                    print(f"関係作成エラー (batch {i}): {e}")
        
        print("段階3: インデックス作成...")
        self.create_indexes()
        
        self.print_statistics()
    
    def print_statistics(self):
        """統計情報表示"""
        mode = "増分処理" if self.incremental else "完全再作成"
        print("\n" + "="*60)
        print(f"{mode}統計:")
        print(f"  総対象ファイル数: {self.stats['total_files']}")
        print(f"  スキップファイル数: {self.stats['skipped_files']}")
        print(f"  処理ファイル数: {self.stats['processed_files']}")
        print(f"  作成特許数: {self.stats['created_patents']}")
        print(f"  作成セクション数: {self.stats['created_sections']}")
        print(f"  構造化セクション数: {self.stats['structured_sections']}")
        print(f"  作成トピック数: {self.stats['created_topics']}")
        print(f"  除外ストップワード数: {self.stats['filtered_stopwords']}")
        print(f"  長文分割処理数: {self.stats['chunked_texts']}")
        print(f"  総チャンク数: {self.stats['total_chunks']}")
        
        if self.stats['total_files'] > 0:
            skip_rate = (self.stats['skipped_files'] / self.stats['total_files']) * 100
            process_rate = (self.stats['processed_files'] / self.stats['total_files']) * 100
            print(f"  スキップ率: {skip_rate:.1f}%")
            print(f"  処理率: {process_rate:.1f}%")
        
        print("="*60)
    
    def close(self):
        """リソースクリーンアップ"""
        self.driver.close()

def main():
    parser = argparse.ArgumentParser(description="構造化セクション対応グラフ作成システム")
    parser.add_argument("--input", "-i", required=True, help="特許JSONファイルディレクトリ")
    parser.add_argument("--stopwords", "-s", help="ストップワードファイル")
    parser.add_argument("--limit", "-l", type=int, default=0, help="処理ファイル数制限")
    parser.add_argument("--workers", "-w", type=int, default=MAX_WORKERS, help="並列処理数")
    parser.add_argument("--rebuild", "-r", action="store_true", help="完全再作成モード（既存データを削除）")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ディレクトリが見つかりません: {args.input}")
        return 1
    
    json_files = list(input_path.rglob("*.json"))
    if not json_files:
        print("JSONファイルが見つかりません")
        return 1
    
    print(f"JSONファイル発見: {len(json_files)}件")
    
    incremental_mode = not args.rebuild
    
    if args.rebuild:
        print("⚠️  完全再作成モード: 既存データを削除して一から作成します")
        confirm = input("続行しますか？ (yes/no): ")
        if confirm.lower() not in ['yes', 'y']:
            print("処理を中止しました")
            return 0
    
    creator = EnhancedLongTextGraphCreator(args.stopwords, args.workers, incremental_mode)
    
    try:
        start_time = time.time()
        
        creator.process_files_smart(json_files, args.limit)
        
        total_time = time.time() - start_time
        processed_files = creator.stats['processed_files']
        
        print(f"\n総処理時間: {total_time:.2f}秒")
        if processed_files > 0:
            print(f"平均処理時間: {total_time/processed_files:.3f}秒/ファイル")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n処理が中断されました")
        return 1
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        creator.close()

if __name__ == "__main__":
    sys.exit(main())