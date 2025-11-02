#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
search_patents.py - 段階1-2の検索（請求項分析強化版・キャッシュ対応）

段階1: ベクトル検索 + 請求項の初期スクリーニングで100件に絞り込み
段階2: グラフ構造分析 + 請求項詳細比較で30件に精密絞り込み
埋め込みキャッシュで2回目以降は超高速
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import re
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase
from tqdm import tqdm
import numpy as np

load_dotenv()

# 環境変数
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "neo4jpass")

# 埋め込みモデル設定
EMBEDDING_MODELS = {
    "small": "text-embedding-3-small",
    "large": "text-embedding-3-large"
}

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY が未設定です")

@dataclass
class ClaimElement:
    """請求項の構成要素"""
    element_id: str
    text: str
    category: str
    keywords: List[str] = field(default_factory=list)
    is_essential: bool = False
    abstraction_level: str = "concrete"

@dataclass
class ClaimAnalysis:
    """請求項分析結果"""
    claim_text: str
    elements: List[ClaimElement]
    essential_elements: List[ClaimElement]
    technical_field: str
    core_concept: str

@dataclass
class PatentSection:
    """特許セクションデータ"""
    section_type: str
    text: str
    embedding: Optional[List[float]] = None

@dataclass
class PatentQuery:
    """特許クエリデータ"""
    pub_id: str
    title: str
    technical_field: Optional[PatentSection] = None
    background_art: Optional[PatentSection] = None
    summary_of_invention: Optional[PatentSection] = None
    tech_problem: Optional[PatentSection] = None
    tech_solution: Optional[PatentSection] = None
    advantageous_effects: Optional[PatentSection] = None
    claims: List[str] = field(default_factory=list)
    claim_analysis: Optional[ClaimAnalysis] = None

@dataclass
class PatentCandidate:
    """特許候補データ（段階1結果）"""
    pub_id: str
    title: str
    ipc_codes: List[str]
    similarity_score: float
    matched_sections: List[Dict[str, str]]
    discovery_method: str
    relevance_path: List[str]
    raw_data: Dict
    initial_claim_score: Optional[float] = None

@dataclass
class ClaimMatchScore:
    """請求項マッチングスコア"""
    element_match_rate: float
    essential_match_rate: float
    concept_similarity: float
    abstraction_compatibility: float
    overall_score: float
    matched_elements: List[str]
    missing_essentials: List[str]

@dataclass
class PatentResult:
    """最終特許検索結果"""
    pub_id: str
    title: str
    ipc_codes: List[str]
    final_score: float
    graph_boost_score: float
    claim_match_score: Optional[ClaimMatchScore]
    matched_sections: List[Dict[str, str]]
    summary: str
    evidence_snippets: List[str]
    relevance_path: List[str]
    graph_analysis: Dict[str, float]
    rank_movement: int

class PatentJSONLoader:
    """特許JSONファイルローダー（請求項専用・キャッシュ版）"""
    
    def __init__(self, json_dir: str, cache_dir: str = "./cache"):
        self.json_dir = Path(json_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.claims_cache = {}
        self.embeddings_cache = {}
        self.file_cache = {}
        
        print(f"\nJSONディレクトリ: {self.json_dir}")
        print(f"キャッシュディレクトリ: {self.cache_dir}")
        
        self.claims_cache_file = self.cache_dir / "claims_cache.json"
        self.embeddings_cache_file = self.cache_dir / "embeddings_cache.pkl"
        
        self._load_cache()
        self._build_claims_index()
        self._save_cache()
    
    def _load_cache(self):
        """キャッシュを読み込み"""
        if self.claims_cache_file.exists():
            with open(self.claims_cache_file, 'r', encoding='utf-8') as f:
                self.claims_cache = json.load(f)
            print(f"請求項キャッシュ読み込み: {len(self.claims_cache)}件")
        
        if self.embeddings_cache_file.exists():
            with open(self.embeddings_cache_file, 'rb') as f:
                self.embeddings_cache = pickle.load(f)
            print(f"埋め込みキャッシュ読み込み: {len(self.embeddings_cache)}件")
    
    def _save_cache(self):
        """キャッシュを保存"""
        with open(self.claims_cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.claims_cache, f, ensure_ascii=False, indent=2)
        
        with open(self.embeddings_cache_file, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)
        
        print(f"キャッシュ保存完了: 請求項{len(self.claims_cache)}件, 埋め込み{len(self.embeddings_cache)}件")
    
    def _build_claims_index(self):
        """請求項1のみをインデックス化（差分のみ）"""
        print("請求項1のインデックス構築中（差分のみ）...")
        
        if not self.json_dir.exists():
            print(f"警告: JSONディレクトリが存在しません: {self.json_dir}")
            return
        
        json_files = list(self.json_dir.rglob("*.json"))
        
        new_count = 0
        for filepath in tqdm(json_files, desc="請求項1インデックス", unit="files"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                pub_id = data.get("bibliographic", {}).get("publication", {}).get("doc_number")
                if not pub_id:
                    continue
                
                self.file_cache[pub_id] = str(filepath)
                
                if pub_id in self.claims_cache:
                    continue
                
                claims = self.extract_claims_from_json(data)
                if claims and len(claims) > 0:
                    self.claims_cache[pub_id] = [claims[0]]
                    new_count += 1
                    
            except Exception:
                continue
        
        print(f"インデックス完了: 新規{new_count}件, 合計{len(self.claims_cache)}件")
    
    def get_claims(self, pub_id: str) -> List[str]:
        """pub_idから請求項1を取得"""
        return self.claims_cache.get(pub_id, [])
    
    def get_embedding(self, pub_id: str) -> Optional[List[float]]:
        """pub_idから埋め込みを取得"""
        return self.embeddings_cache.get(pub_id)
    
    def set_embedding(self, pub_id: str, embedding: List[float]):
        """埋め込みをキャッシュ"""
        self.embeddings_cache[pub_id] = embedding
    
    def extract_claims_from_json(self, data: Dict) -> List[str]:
        """JSONから請求項を抽出"""
        claims = []
        
        if 'claims' in data:
            claims_data = data['claims']
            if isinstance(claims_data, list):
                for claim in claims_data:
                    if isinstance(claim, dict):
                        text = claim.get('text', '')
                        if text:
                            claims.append(text)
                    elif isinstance(claim, str):
                        claims.append(claim)
        
        return claims

class ClaimAnalyzer:
    """請求項分析器（埋め込みベース・バッチ処理版）"""
    
    def __init__(self, openai_client: OpenAI, json_loader):
        self.client = openai_client
        self.embedding_model = "text-embedding-3-small"
        self.json_loader = json_loader
    
    def batch_generate_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """埋め込みをバッチ生成（高速化）"""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="埋め込み生成", unit="batch"):
            batch = texts[i:i+batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                embeddings.extend([data.embedding for data in response.data])
            except Exception as e:
                print(f"警告: バッチ{i}埋め込みエラー: {e}")
                embeddings.extend([[0.0] * 1536] * len(batch))
        
        return embeddings
    
    def precompute_embeddings_for_candidates(self, candidates: List[PatentCandidate]):
        """候補特許の埋め込みを事前計算（キャッシュ活用）"""
        
        texts_to_embed = []
        pub_ids_to_embed = []
        
        for candidate in candidates:
            if self.json_loader.get_embedding(candidate.pub_id) is None:
                claims = self.json_loader.get_claims(candidate.pub_id)
                if claims:
                    texts_to_embed.append(claims[0][:2000])
                    pub_ids_to_embed.append(candidate.pub_id)
        
        if texts_to_embed:
            print(f"埋め込み未キャッシュ: {len(texts_to_embed)}件")
            embeddings = self.batch_generate_embeddings(texts_to_embed)
            
            for pub_id, emb in zip(pub_ids_to_embed, embeddings):
                self.json_loader.set_embedding(pub_id, emb)
        else:
            print("全候補の埋め込みがキャッシュ済み")
    
    def quick_claim_comparison_cached(
        self,
        query_embedding: List[float],
        candidate_pub_id: str
    ) -> float:
        """キャッシュされた埋め込みで高速比較"""
        
        candidate_emb = self.json_loader.get_embedding(candidate_pub_id)
        if not candidate_emb:
            return 0.0
        
        try:
            query_vec = np.array(query_embedding)
            cand_vec = np.array(candidate_emb)
            
            similarity = np.dot(query_vec, cand_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(cand_vec)
            )
            
            return float(similarity)
        except Exception:
            return 0.0
    
    def analyze_claim_simple(self, claim_text: str) -> ClaimAnalysis:
        """請求項を簡易分析（構成要素抽出のみ）"""
        
        elements = self._extract_elements_simple(claim_text)
        essential_elements = [e for e in elements if e.is_essential]
        
        technical_field = self._estimate_technical_field(claim_text)
        core_concept = claim_text[:200] + "..." if len(claim_text) > 200 else claim_text
        
        return ClaimAnalysis(
            claim_text=claim_text,
            elements=elements,
            essential_elements=essential_elements,
            technical_field=technical_field,
            core_concept=core_concept
        )
    
    def _extract_elements_simple(self, claim_text: str) -> List[ClaimElement]:
        """ルールベースで構成要素を抽出"""
        elements = []
        
        patterns = [
            r'([^、。]+)を備え',
            r'([^、。]+)を有し',
            r'([^、。]+)からなり',
            r'([^、。]+)であって',
            r'([^、。]+)において',
        ]
        
        element_texts = []
        for pattern in patterns:
            matches = re.findall(pattern, claim_text)
            element_texts.extend(matches)
        
        if not element_texts:
            element_texts = [s.strip() for s in claim_text.split('、') if len(s.strip()) > 5]
        
        for i, text in enumerate(element_texts[:10]):
            keywords = self._extract_keywords(text)
            is_essential = bool(re.search(r'[一-龯]{2,}', text))
            
            elements.append(ClaimElement(
                element_id=f"E{i+1}",
                text=text[:100],
                category="essential" if is_essential else "optional",
                keywords=keywords,
                is_essential=is_essential,
                abstraction_level="concrete"
            ))
        
        return elements
    
    def _extract_keywords(self, text: str) -> List[str]:
        """キーワード抽出"""
        katakana = re.findall(r'[ァ-ヴー]{2,}', text)
        kanji = re.findall(r'[一-龯]{2,}', text)
        alphanum = re.findall(r'[A-Za-z0-9]{2,}', text)
        
        return (katakana + kanji + alphanum)[:5]
    
    def _estimate_technical_field(self, claim_text: str) -> str:
        """技術分野を推定"""
        if any(word in claim_text for word in ['半導体', 'トランジスタ', '回路']):
            return "半導体・電子回路"
        elif any(word in claim_text for word in ['遊技機', 'スロット', 'パチンコ']):
            return "遊技機"
        elif any(word in claim_text for word in ['表示', 'ディスプレイ', '画面']):
            return "表示装置"
        else:
            return "その他"
    
    def compare_claims_simple(
        self,
        query_analysis: ClaimAnalysis,
        candidate_claims: List[str]
    ) -> ClaimMatchScore:
        """請求項の簡易比較（埋め込み + ルールベース）"""
        
        if not candidate_claims:
            return ClaimMatchScore(
                element_match_rate=0.0,
                essential_match_rate=0.0,
                concept_similarity=0.0,
                abstraction_compatibility=0.0,
                overall_score=0.0,
                matched_elements=[],
                missing_essentials=[]
            )
        
        candidate_claim = candidate_claims[0]
        
        query_keywords = set()
        for elem in query_analysis.elements:
            query_keywords.update(elem.keywords)
        
        candidate_keywords = set(self._extract_keywords(candidate_claim))
        
        if query_keywords:
            keyword_match = len(query_keywords & candidate_keywords) / len(query_keywords)
        else:
            keyword_match = 0.0
        
        essential_keywords = set()
        for elem in query_analysis.essential_elements:
            essential_keywords.update(elem.keywords)
        
        if essential_keywords:
            essential_match = len(essential_keywords & candidate_keywords) / len(essential_keywords)
        else:
            essential_match = 0.0
        
        concept_similarity = keyword_match
        
        overall_score = (
            concept_similarity * 0.5 +
            keyword_match * 0.3 +
            essential_match * 0.2
        )
        
        matched_elements = [
            elem.element_id for elem in query_analysis.elements
            if any(kw in candidate_claim for kw in elem.keywords)
        ]
        
        missing_essentials = [
            elem.element_id for elem in query_analysis.essential_elements
            if not any(kw in candidate_claim for kw in elem.keywords)
        ]
        
        return ClaimMatchScore(
            element_match_rate=keyword_match,
            essential_match_rate=essential_match,
            concept_similarity=concept_similarity,
            abstraction_compatibility=1.0 if concept_similarity > 0.7 else 0.5,
            overall_score=overall_score,
            matched_elements=matched_elements,
            missing_essentials=missing_essentials
        )

class HybridGraphPatentSearchEngine:
    """ハイブリッドグラフ強化特許検索エンジン"""
    
    def __init__(self, json_dir: str):
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        self.embedding_model = None
        self.vector_index_info = None
        
        self.json_loader = PatentJSONLoader(json_dir)
        self.claim_analyzer = ClaimAnalyzer(self.openai_client, self.json_loader)
        
        self._detect_embedding_model()
    
    def close(self):
        """リソースをクリーンアップ"""
        self.neo4j_driver.close()
    
    def _detect_embedding_model(self):
        """データベースのベクトル次元から適切な埋め込みモデルを検出"""
        with self.neo4j_driver.session() as session:
            try:
                result = session.run("""
                    SHOW VECTOR INDEXES
                    YIELD name, labelsOrTypes, properties, options, state
                    WHERE state = "ONLINE"
                    RETURN name, labelsOrTypes, properties, options
                """)
                
                for record in result:
                    if 'sectionEmbeddingIdx' in record['name']:
                        options = record.get('options', {})
                        dimensions = options.get('indexConfig', {}).get('vector.dimensions', 1536)
                        
                        if dimensions == 1536:
                            self.embedding_model = EMBEDDING_MODELS["small"]
                        elif dimensions == 3072:
                            self.embedding_model = EMBEDDING_MODELS["large"]
                        else:
                            self.embedding_model = EMBEDDING_MODELS["small"]
                        
                        print(f"使用埋め込みモデル: {self.embedding_model} (次元数: {dimensions})")
                        
                        self.vector_index_info = {
                            'name': record['name'],
                            'dimensions': dimensions,
                            'available': True
                        }
                        return
                
                print("警告: ベクトルインデックスが見つかりません")
                self.embedding_model = EMBEDDING_MODELS["small"]
                self.vector_index_info = {'available': False}
                
            except Exception as e:
                print(f"警告: インデックス検出エラー: {e}")
                self.embedding_model = EMBEDDING_MODELS["small"]
                self.vector_index_info = {'available': False}
    
    def extract_text_from_section(self, section_obj) -> str:
        """セクションオブジェクトからテキストを抽出"""
        texts = []
        
        def walk(obj, depth=0):
            if depth > 20:
                return
                
            if isinstance(obj, dict):
                if "text" in obj and obj["text"]:
                    text_val = str(obj["text"]).strip()
                    if text_val and text_val not in texts:
                        texts.append(text_val)
                
                for key, val in obj.items():
                    if key != "text":
                        walk(val, depth + 1)
                        
            elif isinstance(obj, list):
                for item in obj:
                    walk(item, depth + 1)
            elif isinstance(obj, str):
                text_val = obj.strip()
                if text_val and text_val not in texts:
                    texts.append(text_val)
        
        walk(section_obj)
        return "\n".join(texts)
    
    def parse_patent_file(self, filepath: str) -> Optional[PatentQuery]:
        """特許JSONファイルを解析してPatentQueryを作成"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            bibl = data.get("bibliographic", {})
            pub_id = bibl.get("publication", {}).get("doc_number", "")
            title = bibl.get("title", "")
            
            if not pub_id:
                print(f"警告: 特許番号が見つかりません: {filepath}")
                return None
            
            claims = []
            if self.json_loader:
                claims = self.json_loader.extract_claims_from_json(data)
            
            if not claims:
                print(f"警告: 請求項が見つかりません: {filepath}")
            else:
                print(f"請求項抽出: {len(claims)}件")
            
            desc = data.get("description", {})
            if not desc:
                print(f"警告: Description セクションが見つかりません: {filepath}")
                return None
            
            sections = {}
            
            for section_key in ["technical-field", "background-art"]:
                if section_key in desc:
                    text = self.extract_text_from_section(desc[section_key])
                    if text:
                        sections[section_key] = PatentSection(section_key, text)
            
            summary_data = desc.get("summary-of-invention")
            
            if summary_data:
                if isinstance(summary_data, list) and summary_data:
                    summary_data = summary_data[0]
                
                if isinstance(summary_data, dict):
                    full_summary_text = self.extract_text_from_section(summary_data)
                    if full_summary_text:
                        sections["summary-of-invention"] = PatentSection("summary-of-invention", full_summary_text)
                    
                    for child_section in ["tech-problem", "tech-solution", "advantageous-effects"]:
                        if child_section in summary_data:
                            text = self.extract_text_from_section(summary_data[child_section])
                            if text:
                                sections[child_section] = PatentSection(child_section, text)
            
            if not sections and not claims:
                print(f"警告: 対象セクションと請求項が見つかりません: {filepath}")
                return None
            
            query = PatentQuery(
                pub_id=pub_id,
                title=title,
                technical_field=sections.get("technical-field"),
                background_art=sections.get("background-art"),
                summary_of_invention=sections.get("summary-of-invention"),
                tech_problem=sections.get("tech-problem"),
                tech_solution=sections.get("tech-solution"),
                advantageous_effects=sections.get("advantageous-effects"),
                claims=claims
            )
            
            print(f"特許解析完了: {pub_id}")
            return query
            
        except Exception as e:
            print(f"エラー: ファイル解析エラー {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def embed_section(self, section: PatentSection) -> bool:
        """セクションテキストを埋め込みベクトルに変換"""
        try:
            if not section.text or len(section.text.strip()) == 0:
                return False
                
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=[section.text]
            )
            section.embedding = response.data[0].embedding
            return True
        except Exception as e:
            print(f"警告: 埋め込み生成エラー ({section.section_type}): {e}")
            return False
    
    def stage1_broad_search_with_claim_filter(
        self, 
        tx, 
        query: PatentQuery, 
        target_k: int = 100
    ) -> List[PatentCandidate]:
        """段階1: ベクトル検索 + 請求項初期スクリーニングで100件に絞り込み"""
        
        print(f"\n{'='*80}")
        print(f"段階1: 広範囲検索 + 請求項初期スクリーニング (目標: {target_k}件)")
        print(f"{'='*80}")
        
        initial_k = target_k * 3
        
        print(f"\nPhase 1: ベクトル検索で初期候補を取得 (目標: {initial_k}件)")
        
        all_results = []
        discovered_patents = {}
        
        min_score_threshold = 0.35
        
        if query.background_art and query.background_art.embedding:
            print(f"  背景技術ベース検索...")
            ba_results = self._vector_search(
                tx, query.background_art.embedding,
                section_filter="background-art", 
                k=initial_k, 
                min_score=min_score_threshold
            )
            
            for result in ba_results:
                if result['pub_id'] not in discovered_patents:
                    result['discovery_method'] = 'background-art-vector'
                    result['relevance_path'] = ['background-art']
                    discovered_patents[result['pub_id']] = result
                    all_results.append(result)
            
            print(f"  {len(ba_results)}件発見 (累計: {len(all_results)}件)")
        
        if query.summary_of_invention and query.summary_of_invention.embedding:
            print(f"  発明概要ベース検索...")
            si_results = self._vector_search(
                tx, query.summary_of_invention.embedding,
                section_filter="summary-of-invention", 
                k=initial_k, 
                min_score=min_score_threshold
            )
            
            for result in si_results:
                if result['pub_id'] not in discovered_patents:
                    result['discovery_method'] = 'summary-invention-vector'
                    result['relevance_path'] = ['summary-of-invention']
                    discovered_patents[result['pub_id']] = result
                    all_results.append(result)
                else:
                    existing = discovered_patents[result['pub_id']]
                    existing['score'] = max(existing['score'], result['score'])
            
            new_count = len([r for r in si_results if r['pub_id'] not in [ar['pub_id'] for ar in all_results[:len(ba_results)] if 'ba_results' in locals()]])
            print(f"  新規発見 (累計: {len(all_results)}件)")
        
        structured_sections = {
            'tech-solution': query.tech_solution,
            'tech-problem': query.tech_problem,
        }
        
        for section_name, section_obj in structured_sections.items():
            if section_obj and section_obj.embedding:
                print(f"  {section_name}検索...")
                struct_results = self._vector_search(
                    tx, section_obj.embedding,
                    section_filter=section_name, 
                    k=initial_k // 2, 
                    min_score=min_score_threshold
                )
                
                for result in struct_results:
                    if result['pub_id'] not in discovered_patents:
                        result['discovery_method'] = f'{section_name}-vector'
                        result['relevance_path'] = [section_name]
                        discovered_patents[result['pub_id']] = result
                        all_results.append(result)
        
        print(f"\nベクトル検索完了: {len(all_results)}件の初期候補")
        
        candidates = self._convert_to_candidates(all_results)
        
        if query.claims and query.claim_analysis and self.json_loader:
            print(f"\nPhase 2: 請求項による初期スクリーニング")
            print(f"  クエリ核心概念: {query.claim_analysis.core_concept[:100]}...")
            
            print("  クエリ請求項の埋め込み生成中...")
            query_embedding = self.claim_analyzer.batch_generate_embeddings([query.claims[0][:2000]])[0]
            
            print("  候補請求項の埋め込み生成中...")
            self.claim_analyzer.precompute_embeddings_for_candidates(candidates)
            
            print("  類似度計算中...")
            candidates_with_claim_score = []
            
            def process_candidate_claim(candidate):
                claim_score = self.claim_analyzer.quick_claim_comparison_cached(
                    query_embedding,
                    candidate.pub_id
                )
                candidate.initial_claim_score = claim_score
                return candidate
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_candidate_claim, c) for c in candidates]
                
                for future in tqdm(as_completed(futures), 
                                 total=len(candidates), 
                                 desc="請求項スクリーニング", 
                                 unit="特許"):
                    result = future.result()
                    candidates_with_claim_score.append(result)
            
            for candidate in candidates_with_claim_score:
                if candidate.initial_claim_score is not None:
                    candidate.similarity_score = (
                        candidate.similarity_score * 0.6 +
                        candidate.initial_claim_score * 0.4
                    )
            
            candidates = candidates_with_claim_score
            
            print(f"  請求項スコアを算出")
        
        candidates.sort(key=lambda x: x.similarity_score, reverse=True)
        final_candidates = candidates[:target_k]
        
        print(f"\n{'='*80}")
        print(f"段階1完了: {len(final_candidates)}件に絞り込み (目標: {target_k}件)")
        print(f"{'='*80}\n")
        
        return final_candidates
    
    def _vector_search(
        self, 
        tx, 
        embedding: List[float], 
        section_filter: str = None, 
        k: int = 50, 
        min_score: float = 0.4
    ) -> List[Dict]:
        """ベクトル検索の実行"""
        if not self.vector_index_info or not self.vector_index_info.get('available', False):
            return []
        
        try:
            if section_filter:
                query = """
                WITH $qv AS qv
                CALL db.index.vector.queryNodes('sectionEmbeddingIdx', $k, qv)
                YIELD node, score
                WITH node, score
                WHERE score >= $min_score AND node.section_type = $section_type
                
                RETURN 
                    node.node_id AS node_id,
                    node.section_type AS section_type,
                    node.pub_id AS pub_id,
                    substring(coalesce(node.text, ''), 0, 500) AS text,
                    score
                ORDER BY score DESC
                LIMIT $k
                """
                result = tx.run(query, qv=embedding, k=k*3, section_type=section_filter, min_score=min_score)
            else:
                query = """
                WITH $qv AS qv
                CALL db.index.vector.queryNodes('sectionEmbeddingIdx', $k, qv)
                YIELD node, score
                WITH node, score
                WHERE score >= $min_score
                
                RETURN 
                    node.node_id AS node_id,
                    node.section_type AS section_type,
                    node.pub_id AS pub_id,
                    substring(coalesce(node.text, ''), 0, 500) AS text,
                    score
                ORDER BY score DESC
                LIMIT $k
                """
                result = tx.run(query, qv=embedding, k=k*3, min_score=min_score)
            
            return result.data()
            
        except Exception as e:
            print(f"警告: ベクトル検索エラー: {e}")
            return []
    
    def _convert_to_candidates(self, results: List[Dict]) -> List[PatentCandidate]:
        """検索結果をPatentCandidateに変換"""
        candidates = []
        
        for result in results:
            candidate = PatentCandidate(
                pub_id=result['pub_id'],
                title="",
                ipc_codes=[],
                similarity_score=result['score'],
                matched_sections=[{
                    'section_type': result['section_type'],
                    'text': result['text']
                }],
                discovery_method=result.get('discovery_method', 'unknown'),
                relevance_path=result.get('relevance_path', []),
                raw_data=result
            )
            candidates.append(candidate)
        
        return candidates
    
    def stage2_graph_and_detailed_claim_analysis(
        self, 
        tx, 
        candidates: List[PatentCandidate], 
        query: PatentQuery, 
        final_k: int = 30
    ) -> List[PatentResult]:
        """段階2: グラフ構造 + 請求項詳細比較で30件に精密絞り込み"""
        
        print(f"\n{'='*80}")
        print(f"段階2: グラフ構造分析 + 請求項詳細比較 (目標: {final_k}件)")
        print(f"{'='*80}")
        
        query_claim_analysis = query.claim_analysis
        if not query_claim_analysis and query.claims:
            print("\nクエリ特許の請求項を分析中...")
            query_claim_analysis = self.claim_analyzer.analyze_claim_simple(query.claims[0])
            query.claim_analysis = query_claim_analysis
            
            print(f"  構成要素: {len(query_claim_analysis.elements)}個")
            print(f"  必須要件: {len(query_claim_analysis.essential_elements)}個")
        
        candidate_ids = [c.pub_id for c in candidates]
        
        print(f"\nグラフ構造分析中...")
        patent_details = self._get_patent_details_batch(tx, candidate_ids)
        
        graph_analyses = {}
        
        with tqdm(total=5, desc="グラフ分析", unit="種類") as pbar:
            ipc_analysis = self._analyze_ipc_relationships(tx, candidates, query)
            pbar.update(1)
            
            citation_analysis = self._analyze_citation_networks(tx, candidates)
            pbar.update(1)
            
            structural_analysis = self._analyze_structural_similarity(tx, candidates, query)
            pbar.update(1)
            
            section_analysis = self._analyze_section_relationships(tx, candidates, query)
            pbar.update(1)
            
            temporal_analysis = self._analyze_temporal_patterns(tx, candidates)
            pbar.update(1)
        
        for candidate in candidates:
            pub_id = candidate.pub_id
            
            graph_analyses[pub_id] = {
                'ipc_boost': ipc_analysis.get(pub_id, 0.0),
                'citation_boost': citation_analysis.get(pub_id, 0.0),
                'structural_boost': structural_analysis.get(pub_id, 0.0),
                'section_boost': section_analysis.get(pub_id, 0.0),
                'temporal_boost': temporal_analysis.get(pub_id, 0.0)
            }
        
        print(f"\n候補特許の請求項詳細分析中...")
        claim_match_scores = {}
        
        if query_claim_analysis and self.json_loader:
            def process_detailed_claim(candidate):
                pub_id = candidate.pub_id
                candidate_claims = self.json_loader.get_claims(pub_id)
                
                if candidate_claims:
                    claim_score = self.claim_analyzer.compare_claims_simple(
                        query_claim_analysis,
                        candidate_claims
                    )
                    return (pub_id, claim_score)
                return (pub_id, None)
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(process_detailed_claim, c) for c in candidates]
                
                for future in tqdm(as_completed(futures), 
                                 total=len(candidates), 
                                 desc="請求項詳細比較", 
                                 unit="特許"):
                    pub_id, claim_score = future.result()
                    if claim_score:
                        claim_match_scores[pub_id] = claim_score
        
        print(f"\n最終スコア計算中...")
        final_results = []
        
        for candidate in tqdm(candidates, desc="最終スコア計算", unit="特許"):
            pub_id = candidate.pub_id
            patent_detail = patent_details.get(pub_id, {})
            graph_analysis = graph_analyses.get(pub_id, {})
            claim_match_score = claim_match_scores.get(pub_id)
            
            graph_boost = (
                graph_analysis.get('ipc_boost', 0) * 0.25 +
                graph_analysis.get('citation_boost', 0) * 0.20 +
                graph_analysis.get('structural_boost', 0) * 0.20 +
                graph_analysis.get('section_boost', 0) * 0.25 +
                graph_analysis.get('temporal_boost', 0) * 0.10
            )
            
            if claim_match_score:
                final_score = (
                    candidate.similarity_score * 0.2 +
                    graph_boost * 0.2 +
                    claim_match_score.overall_score * 0.6
                )
            else:
                final_score = (
                    candidate.similarity_score * 0.5 +
                    graph_boost * 0.5
                )
            
            summary = self._generate_patent_summary(patent_detail, candidate.matched_sections)
            
            result = PatentResult(
                pub_id=pub_id,
                title=patent_detail.get('title', candidate.title),
                ipc_codes=patent_detail.get('ipc_codes', candidate.ipc_codes),
                final_score=final_score,
                graph_boost_score=graph_boost,
                claim_match_score=claim_match_score,
                matched_sections=candidate.matched_sections,
                summary=summary,
                evidence_snippets=[s['text'][:150] for s in candidate.matched_sections[:2]],
                relevance_path=candidate.relevance_path,
                graph_analysis=graph_analysis,
                rank_movement=0
            )
            
            final_results.append(result)
        
        final_results.sort(key=lambda x: x.final_score, reverse=True)
        top_results = final_results[:final_k]
        
        for new_rank, result in enumerate(top_results):
            original_rank = next(
                (i for i, c in enumerate(candidates) if c.pub_id == result.pub_id),
                new_rank
            )
            result.rank_movement = original_rank - new_rank
        
        print(f"\n{'='*80}")
        print(f"段階2完了: {len(top_results)}件に絞り込み (目標: {final_k}件)")
        print(f"{'='*80}\n")
        
        return top_results
    
    def _get_patent_details_batch(self, tx, pub_ids: List[str]) -> Dict[str, Dict]:
        """特許詳細情報のバッチ取得"""
        if not pub_ids:
            return {}
        
        query = """
        UNWIND $pub_ids AS pub_id
        MATCH (p:Patent {pub_id: pub_id})
        OPTIONAL MATCH (p)-[:HAS_SECTION]->(s:Section)
        WHERE s.section_type IN ['technical-field', 'background-art', 'summary-of-invention',
                                  'tech-problem', 'tech-solution', 'advantageous-effects']
        
        RETURN 
            p.pub_id AS pub_id,
            coalesce(p.title, '不明') AS title,
            coalesce(p.ipc_codes, []) AS ipc_codes,
            collect({
                section_type: coalesce(s.section_type, ''),
                text: substring(coalesce(s.text, ''), 0, 300)
            }) AS sections
        """
        
        result = tx.run(query, pub_ids=pub_ids)
        patents = {}
        
        for record in result:
            patents[record['pub_id']] = {
                'title': record['title'],
                'ipc_codes': record['ipc_codes'],
                'sections': record['sections']
            }
        
        return patents
    
    def _analyze_ipc_relationships(self, tx, candidates: List[PatentCandidate], query: PatentQuery) -> Dict[str, float]:
        candidate_ids = [c.pub_id for c in candidates]
        query_ipcs = self._estimate_query_ipcs(tx, query)
        
        if not query_ipcs:
            return {}
        
        ipc_query = """
        UNWIND $candidate_ids AS pub_id
        MATCH (p:Patent {pub_id: pub_id})
        WITH p, [ipc IN p.ipc_codes WHERE ipc IN $query_ipcs] AS matching_ipcs
        WITH p, toFloat(size(matching_ipcs)) / size($query_ipcs) AS ipc_similarity
        WHERE ipc_similarity > 0
        RETURN p.pub_id AS pub_id, ipc_similarity * 0.15 AS ipc_boost
        """
        
        result = tx.run(ipc_query, candidate_ids=candidate_ids, query_ipcs=query_ipcs)
        return {record['pub_id']: record['ipc_boost'] for record in result}
    
    def _analyze_citation_networks(self, tx, candidates: List[PatentCandidate]) -> Dict[str, float]:
        candidate_ids = [c.pub_id for c in candidates]
        
        citation_query = """
        UNWIND $candidate_ids AS pub_id
        MATCH (p:Patent {pub_id: pub_id})
        OPTIONAL MATCH (p)-[:CITES]->(cited:Patent)
        OPTIONAL MATCH (p)<-[:CITES]-(citing:Patent)
        WITH p,
             count(DISTINCT cited) AS out_citations,
             count(DISTINCT citing) AS in_citations
        WITH p,
             toFloat(in_citations) / (out_citations + 1.0) AS citation_importance
        RETURN p.pub_id AS pub_id, citation_importance * 0.05 AS citation_boost
        """
        
        result = tx.run(citation_query, candidate_ids=candidate_ids)
        return {record['pub_id']: record['citation_boost'] for record in result if record['citation_boost'] > 0}
    
    def _analyze_structural_similarity(self, tx, candidates: List[PatentCandidate], query: PatentQuery) -> Dict[str, float]:
        candidate_ids = [c.pub_id for c in candidates]
        
        query_sections = []
        for attr in ['technical_field', 'background_art', 'summary_of_invention', 
                     'tech_problem', 'tech_solution', 'advantageous_effects']:
            section_obj = getattr(query, attr, None)
            if section_obj:
                section_name = attr.replace('_', '-')
                query_sections.append(section_name)
        
        if not query_sections:
            return {}
            
        structural_query = """
        UNWIND $candidate_ids AS pub_id
        MATCH (p:Patent {pub_id: pub_id})-[:HAS_SECTION]->(s:Section)
        WITH p, collect(DISTINCT s.section_type) AS candidate_sections
        WITH p, 
             [section IN candidate_sections WHERE section IN $query_sections] AS common_sections,
             candidate_sections
        WITH p,
             toFloat(size(common_sections)) / size($query_sections) AS section_coverage
        WHERE section_coverage > 0.3
        RETURN p.pub_id AS pub_id, section_coverage * 0.10 AS structural_boost
        """
        
        result = tx.run(structural_query, candidate_ids=candidate_ids, query_sections=query_sections)
        return {record['pub_id']: record['structural_boost'] for record in result}
    
    def _analyze_section_relationships(self, tx, candidates: List[PatentCandidate], query: PatentQuery) -> Dict[str, float]:
        candidate_ids = [c.pub_id for c in candidates]
        
        query_keywords = []
        for attr in ['technical_field', 'tech_solution', 'summary_of_invention']:
            section_obj = getattr(query, attr, None)
            if section_obj:
                query_keywords.extend(self._extract_technical_keywords(section_obj.text)[:5])
        
        if not query_keywords:
            return {}
        
        section_query = """
        UNWIND $candidate_ids AS pub_id
        MATCH (p:Patent {pub_id: pub_id})-[:HAS_SECTION]->(s:Section)
        WITH p, s, 
             [keyword IN $query_keywords WHERE toLower(s.text) CONTAINS toLower(keyword)] AS matched_keywords
        WITH p, 
             collect({keyword_count: size(matched_keywords)}) AS section_matches
        WITH p, 
             reduce(total = 0, match IN section_matches | total + match.keyword_count) AS total_matches
        WHERE total_matches > 0
        RETURN p.pub_id AS pub_id, toFloat(total_matches) / size($query_keywords) * 0.10 AS section_boost
        """
        
        result = tx.run(section_query, candidate_ids=candidate_ids, query_keywords=query_keywords)
        return {record['pub_id']: min(record['section_boost'], 0.15) for record in result}
    
    def _analyze_temporal_patterns(self, tx, candidates: List[PatentCandidate]) -> Dict[str, float]:
        candidate_ids = [c.pub_id for c in candidates]
        
        temporal_query = """
        UNWIND $candidate_ids AS pub_id
        MATCH (p:Patent {pub_id: pub_id})
        WHERE p.publication_date IS NOT NULL
        WITH p, date(p.publication_date) AS pub_date
        WITH p,
             CASE 
                 WHEN pub_date > date('2020-01-01') THEN 0.03
                 WHEN pub_date > date('2018-01-01') THEN 0.02
                 WHEN pub_date > date('2015-01-01') THEN 0.01
                 ELSE 0.0
             END AS recency_boost
        RETURN p.pub_id AS pub_id, recency_boost AS temporal_boost
        """
        
        result = tx.run(temporal_query, candidate_ids=candidate_ids)
        return {record['pub_id']: record['temporal_boost'] for record in result if record['temporal_boost'] > 0}
    
    def _estimate_query_ipcs(self, tx, query: PatentQuery) -> List[str]:
        technical_keywords = []
        if query.technical_field:
            technical_keywords.extend(self._extract_technical_keywords(query.technical_field.text)[:3])
        
        if not technical_keywords:
            return []
        
        ipc_query = """
        MATCH (p:Patent)-[:HAS_SECTION]->(s:Section)
        WHERE any(keyword IN $keywords WHERE toLower(s.text) CONTAINS toLower(keyword))
        WITH p.ipc_codes AS ipcs
        UNWIND ipcs AS ipc
        RETURN ipc, count(*) AS frequency
        ORDER BY frequency DESC
        LIMIT 5
        """
        
        result = tx.run(ipc_query, keywords=technical_keywords)
        return [record['ipc'] for record in result]
    
    def _extract_technical_keywords(self, text: str) -> List[str]:
        katakana_terms = re.findall(r'[ァ-ヴー]{3,}', text)
        kanji_terms = re.findall(r'[一-龯]{3,8}', text)
        alphanumeric_terms = re.findall(r'[A-Za-z0-9]{2,}', text)
        
        all_candidates = katakana_terms + kanji_terms + alphanumeric_terms
        term_freq = Counter(all_candidates)
        
        filtered_terms = []
        for term, freq in term_freq.items():
            if len(term) >= 3 and freq >= 1:
                if term not in ['について', 'により', 'として', 'において', 'することが']:
                    filtered_terms.append(term)
        
        return filtered_terms[:20]
    
    def _generate_patent_summary(self, patent_detail: Dict, matched_sections: List[Dict]) -> str:
        title = patent_detail.get('title', '')
        ipc_codes = patent_detail.get('ipc_codes', [])
        
        if not title:
            return "要約生成不可"
        
        summary_parts = [f"特許: {title}"]
        
        if ipc_codes:
            summary_parts.append(f"分類: {', '.join(ipc_codes[:2])}")
        
        if matched_sections:
            section_types = [s.get('section_type', '') for s in matched_sections]
            summary_parts.append(f"マッチ: {', '.join(set(section_types))}")
        
        return " | ".join(summary_parts)
    
    def search_similar_patents_from_file(
        self, 
        filepath: str, 
        broad_k: int = 100,
        final_k: int = 30
    ) -> List[PatentResult]:
        """ハイブリッド検索のメイン処理"""
        
        print(f"\n{'='*80}")
        print(f"特許検索システム起動")
        print(f"{'='*80}")
        print(f"入力ファイル: {filepath}")
        print(f"段階1目標: {broad_k}件")
        print(f"段階2目標: {final_k}件")
        
        query = self.parse_patent_file(filepath)
        if not query:
            print("エラー: ファイル解析に失敗しました")
            return []
        
        print(f"\n解析完了")
        print(f"  特許番号: {query.pub_id}")
        print(f"  タイトル: {query.title}")
        
        if self.vector_index_info and self.vector_index_info.get('available', False):
            print(f"\n埋め込みベクトル生成中...")
            
            sections_to_embed = [
                ('technical_field', '技術分野'),
                ('background_art', '背景技術'),
                ('summary_of_invention', '発明概要'),
                ('tech_problem', '課題'),
                ('tech_solution', '解決手段'),
                ('advantageous_effects', '効果')
            ]
            
            sections_embedded = 0
            for attr_name, display_name in sections_to_embed:
                section_obj = getattr(query, attr_name, None)
                if section_obj:
                    if self.embed_section(section_obj):
                        print(f"  {display_name}: {len(section_obj.text)}文字")
                        sections_embedded += 1
            
            if sections_embedded == 0:
                print("エラー: 埋め込み生成に失敗しました")
                return []
        
        if query.claims:
            print(f"\nクエリ請求項の事前分析...")
            query.claim_analysis = self.claim_analyzer.analyze_claim_simple(query.claims[0])
            print(f"  構成要素: {len(query.claim_analysis.elements)}個")
            print(f"  必須要件: {len(query.claim_analysis.essential_elements)}個")
        
        with self.neo4j_driver.session() as session:
            try:
                candidates = session.execute_read(
                    self.stage1_broad_search_with_claim_filter, 
                    query, 
                    broad_k
                )
                
                if not candidates:
                    print("エラー: 段階1で候補が見つかりませんでした")
                    return []
                
                final_results = session.execute_read(
                    self.stage2_graph_and_detailed_claim_analysis,
                    candidates,
                    query,
                    final_k
                )
                
            except Exception as e:
                print(f"エラー: 検索処理エラー: {e}")
                import traceback
                traceback.print_exc()
                return []
        
        return final_results
    
    def format_results(self, results: List[PatentResult], output_format: str = "console") -> str:
        """検索結果をフォーマット"""
        if not results:
            return "類似特許が見つかりませんでした"
        
        if output_format == "json":
            return json.dumps([{
                'rank': i+1,
                'pub_id': r.pub_id,
                'title': r.title,
                'ipc_codes': r.ipc_codes,
                'final_score': round(r.final_score, 4),
                'graph_boost': round(r.graph_boost_score, 4),
                'claim_match': {
                    'overall_score': round(r.claim_match_score.overall_score, 4) if r.claim_match_score else None,
                    'essential_match_rate': round(r.claim_match_score.essential_match_rate, 4) if r.claim_match_score else None,
                    'matched_elements': r.claim_match_score.matched_elements if r.claim_match_score else [],
                    'missing_essentials': r.claim_match_score.missing_essentials if r.claim_match_score else []
                } if r.claim_match_score else None,
                'rank_movement': r.rank_movement,
                'matched_sections': [{'type': s['section_type'], 'preview': s['text'][:100]} for s in r.matched_sections],
                'graph_analysis': {k: round(v, 4) for k, v in r.graph_analysis.items()}
            } for i, r in enumerate(results)], ensure_ascii=False, indent=2)
        
        output = []
        output.append(f"\n{'='*80}")
        output.append(f"最終検索結果: {len(results)}件")
        output.append(f"{'='*80}\n")
        
        for i, result in enumerate(results, 1):
            output.append(f"[{i:02d}] {result.pub_id}")
            output.append(f"タイトル: {result.title}")
            
            if result.ipc_codes:
                output.append(f"IPC: {', '.join(result.ipc_codes[:3])}")
            
            output.append(f"最終スコア: {result.final_score:.4f}")
            
            if result.claim_match_score:
                cms = result.claim_match_score
                output.append(f"請求項分析:")
                output.append(f"  総合スコア: {cms.overall_score:.3f}")
                output.append(f"  必須カバー率: {cms.essential_match_rate:.3f}")
                output.append(f"  要素一致率: {cms.element_match_rate:.3f}")
                output.append(f"  概念類似度: {cms.concept_similarity:.3f}")
            
            if result.rank_movement > 0:
                output.append(f"ランク上昇: +{result.rank_movement}位")
            elif result.rank_movement < 0:
                output.append(f"ランク下降: {result.rank_movement}位")
            
            output.append("-" * 80)
        
        return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description="ハイブリッドグラフ強化特許検索システム")
    parser.add_argument("--file", "-f", required=True, help="入力特許JSONファイルパス")
    parser.add_argument("--json-dir", required=True, help="特許JSONファイルディレクトリ")
    parser.add_argument("--broad-k", type=int, default=100, help="段階1の候補数")
    parser.add_argument("--final-k", type=int, default=30, help="段階2の最終結果数")
    parser.add_argument("--format", choices=["console", "json"], default="json", help="出力フォーマット")
    parser.add_argument("--output", "-o", help="JSON出力ファイルパス")
    
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        print(f"エラー: ファイルが見つかりません: {args.file}")
        return 1
    
    if not Path(args.json_dir).exists():
        print(f"エラー: JSONディレクトリが見つかりません: {args.json_dir}")
        return 1
    
    search_engine = HybridGraphPatentSearchEngine(json_dir=args.json_dir)
    
    try:
        results = search_engine.search_similar_patents_from_file(
            filepath=args.file,
            broad_k=args.broad_k,
            final_k=args.final_k
        )
        
        formatted_output = search_engine.format_results(results, args.format)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            print(f"\n結果を保存しました: {args.output}")
        else:
            print(formatted_output)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n検索が中断されました")
        return 1
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        search_engine.close()

if __name__ == "__main__":
    sys.exit(main())