#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3: Claims-based Novelty Assessment System (JSON+Graph Hybrid) - Fixed Version
請求項ベース新規性評価システム（pub_id形式修正版）
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available, using fallback similarity calculation")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

load_dotenv()

# 環境変数
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASS", "neo4jpass")

# JSONファイルディレクトリ（環境変数または引数で指定）
JSON_DATA_DIR = os.environ.get("PATENT_JSON_DIR", "./patent_data")

# ログ設定
def setup_logging(log_file: str = None, verbose: bool = False):
    """ログ設定"""
    log_level = logging.DEBUG if verbose else logging.INFO

    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# データクラス
@dataclass
class ClaimElement:
    """請求項の構成要素"""
    element_id: str
    text: str
    category: str  # structure, function, effect
    keywords: List[str] = field(default_factory=list)

@dataclass
class PatentContent:
    """特許の完全な内容"""
    pub_id: str
    title: str
    claims: List[str]
    claim_elements: List[ClaimElement]
    problem: str
    solution: str
    effect: str
    embodiments: List[str]
    ipc_codes: List[str]
    priority_date: Optional[str] = None
    file_path: Optional[str] = None  # JSONファイルパス
    claims_embedding: Optional[List[float]] = field(default=None)
    problem_embedding: Optional[List[float]] = field(default=None)
    solution_embedding: Optional[List[float]] = field(default=None)

@dataclass
class ElementMapping:
    """構成要素のマッピング"""
    query_element: ClaimElement
    candidate_element: Optional[ClaimElement]
    is_disclosed: bool
    disclosure_type: str  # explicit, implicit, equivalent, not_found
    evidence: str

@dataclass
class ClaimComparison:
    """請求項比較結果"""
    all_elements_disclosed: bool
    element_mappings: List[ElementMapping]
    combination_disclosed: bool
    functional_equivalence: bool
    detailed_analysis: str

@dataclass
class GraphAnalysis:
    """グラフ分析結果"""
    citation_paths: int
    min_citation_distance: int
    common_inventors: int
    common_ipc_exact: int
    common_ipc_group: int
    common_keywords: List[str]
    relatedness_score: float

@dataclass
class NoveltyAssessment:
    """新規性評価結果"""
    pub_id: str
    title: str
    claim_comparison: ClaimComparison
    technical_similarity: float
    graph_analysis: GraphAnalysis
    ipc_similarity: float
    problem_solution_match: Dict[str, bool]
    embodiment_support: bool
    priority_valid: bool
    novelty_risk_level: str
    blocking_power: float
    confidence_score: float
    detailed_reason: str
    evidence_snippets: List[str]

@dataclass
class FinalRanking:
    """最終ランキング結果"""
    rank: int
    pub_id: str
    title: str
    ipc_codes: List[str]
    blocking_power: float
    claim_match_rate: float
    graph_score: float
    assessment: NoveltyAssessment
    original_rank: int
    rank_change: int

# JSONファイルキャッシュ
class PatentJSONCache:
    """特許JSONファイルキャッシュ"""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.cache = {}
        self.pub_id_to_path = {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # インデックス構築
        self._build_index()

    def _build_index(self):
        """pub_idとファイルパスのインデックス構築"""
        self.logger.info(f"JSONファイルインデックス構築中: {self.base_dir}")

        json_files = list(self.base_dir.rglob("*.json"))

        for filepath in tqdm(json_files, desc="インデックス構築", disable=not TQDM_AVAILABLE):
            try:
                pub_id = self._extract_pub_id(filepath)
                if pub_id:
                    self.pub_id_to_path[pub_id] = str(filepath)
            except Exception as e:
                self.logger.debug(f"ファイル読み込みスキップ: {filepath} - {e}")

        self.logger.info(f"インデックス完了: {len(self.pub_id_to_path)}件の特許")

    def _extract_pub_id(self, filepath: Path) -> Optional[str]:
        """ファイルからpub_id抽出（軽量版）"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                # 最初の部分だけ読んでpub_id探索（高速化）
                content = f.read(10000)  # 最初の10KB
                if '"doc_number"' in content:
                    data = json.load(open(filepath, 'r', encoding='utf-8'))
                    return data.get("bibliographic", {}).get("publication", {}).get("doc_number")
        except:
            return None
        return None

    def get_patent_data(self, pub_id: str) -> Optional[Dict]:
        """pub_idから特許データ取得"""
        # キャッシュ確認
        if pub_id in self.cache:
            return self.cache[pub_id]

        # ファイルパス取得
        filepath = self.pub_id_to_path.get(pub_id)
        if not filepath:
            self.logger.warning(f"特許ファイルが見つかりません: {pub_id}")
            return None

        # ファイル読み込み
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # キャッシュに格納
            self.cache[pub_id] = data
            return data

        except Exception as e:
            self.logger.error(f"ファイル読み込みエラー {filepath}: {e}")
            return None

    def clear_cache(self):
        """キャッシュクリア（メモリ管理）"""
        self.cache.clear()

# メインエンジン
class HybridNoveltyEngine:
    """ハイブリッド新規性評価エンジン（JSON+Graph）"""

    def __init__(self, json_dir: str = JSON_DATA_DIR):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        self.json_cache = PatentJSONCache(json_dir)
        self.embedding_model = "text-embedding-3-small"

        self.stats = {
            'patents_analyzed': 0,
            'claims_extracted': 0,
            'json_cache_hits': 0,
            'json_file_reads': 0,
            'graph_queries': 0,
            'graph_hits': 0,  # グラフDBでヒットした数を追加
            'llm_calls': 0,
            'embeddings_generated': 0,
            'errors': 0
        }

        self.logger.info(f"ハイブリッドエンジン初期化完了")
        self.logger.info(f"JSONディレクトリ: {json_dir}")
        self.logger.info(f"利用可能特許数: {len(self.json_cache.pub_id_to_path)}")

        # グラフDBの状態確認
        self._check_graph_db_status()

    def _check_graph_db_status(self):
        """グラフDBの状態を確認"""
        try:
            with self.neo4j_driver.session() as session:
                result = session.run("""
                    MATCH (p:Patent)
                    WITH count(p) as patent_count
                    MATCH (s:Section) WHERE s.embedding IS NOT NULL
                    RETURN patent_count, count(s) as section_count
                """).single()

                if result:
                    self.logger.info(f"グラフDB状態: 特許{result['patent_count']}件, セクション{result['section_count']}件")

                    # サンプルpub_idを取得して形式確認
                    sample = session.run("MATCH (p:Patent) RETURN p.pub_id as pub_id LIMIT 3")
                    sample_ids = [r['pub_id'] for r in sample]
                    if sample_ids:
                        self.logger.info(f"グラフDB内のpub_id形式サンプル: {sample_ids}")
                else:
                    self.logger.warning("グラフDBが空です")
        except Exception as e:
            self.logger.error(f"グラフDB接続エラー: {e}")

    def close(self):
        self.logger.info(f"統計: {self.stats}")
        self.neo4j_driver.close()

    def run_analysis(
        self,
        stage2_json: str,
        query_file: str,
        output_file: str = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """メイン分析処理"""

        self.logger.info("="*80)
        self.logger.info("ハイブリッド新規性評価システム起動")
        self.logger.info("="*80)

        start_time = time.time()

        # Stage2結果読み込み
        with open(stage2_json, 'r', encoding='utf-8') as f:
            stage2_data = json.load(f)

        if isinstance(stage2_data, list):
            candidates = stage2_data[:30]
        elif isinstance(stage2_data, dict):
            candidates = stage2_data.get("results", stage2_data.get("data", []))[:30]
        else:
            self.logger.error(f"不明なStage2形式: {type(stage2_data)}")
            return {}

        self.logger.info(f"候補特許数: {len(candidates)}")

        # Phase 1: クエリ特許の解析
        self.logger.info("\n【Phase 1: クエリ特許解析】")
        query_content = self._extract_patent_content_from_json(query_file, is_file=True)

        if not query_content:
            self.logger.error("クエリ特許の解析失敗")
            return {}

        self._log_patent_content("クエリ特許", query_content)

        # Phase 2: 候補特許の評価
        self.logger.info("\n【Phase 2: 候補特許評価】")
        assessments = []

        # メモリ管理のため、バッチ処理
        batch_size = 10
        for batch_start in range(0, len(candidates), batch_size):
            batch_end = min(batch_start + batch_size, len(candidates))
            batch_candidates = candidates[batch_start:batch_end]

            self.logger.info(f"バッチ処理 {batch_start+1}-{batch_end}/{len(candidates)}")

            for candidate in tqdm(batch_candidates, desc="特許解析") if TQDM_AVAILABLE else batch_candidates:
                pub_id = candidate.get('pub_id', '')
                if not pub_id:
                    continue

                # JSONから特許内容取得
                cand_content = self._extract_patent_content_from_json(pub_id)

                if not cand_content:
                    self.logger.warning(f"  解析失敗: {pub_id}")
                    continue

                # 新規性評価
                assessment = self._assess_novelty(query_content, cand_content)
                assessments.append(assessment)

                self.logger.debug(f"  {pub_id}: 否定力={assessment.blocking_power:.3f}, グラフスコア={assessment.graph_analysis.relatedness_score:.3f}")

            # バッチごとにキャッシュクリア（メモリ管理）
            self.json_cache.clear_cache()

        if not assessments:
            self.logger.error("評価可能な候補が0件でした")
            return {}

        # 統計情報出力
        self.logger.info(f"\nグラフDB分析統計: {self.stats['graph_hits']}/{self.stats['graph_queries']}件ヒット")

        # 最終ランキング
        assessments.sort(key=lambda x: x.blocking_power, reverse=True)
        top_results = assessments[:top_k]

        # 結果整形
        final_rankings = []
        for rank, assessment in enumerate(top_results, 1):
            original_rank = next((i for i, c in enumerate(candidates)
                                 if c.get('pub_id') == assessment.pub_id), -1)

            final_rankings.append(FinalRanking(
                rank=rank,
                pub_id=assessment.pub_id,
                title=assessment.title,
                ipc_codes=next((c.get('ipc_codes', []) for c in candidates
                              if c.get('pub_id') == assessment.pub_id), []),
                blocking_power=assessment.blocking_power,
                claim_match_rate=self._calculate_match_rate(assessment),
                graph_score=assessment.graph_analysis.relatedness_score,
                assessment=assessment,
                original_rank=original_rank + 1 if original_rank >= 0 else 999,
                rank_change=(original_rank + 1 - rank) if original_rank >= 0 else 0
            ))

        # 結果記録
        all_results = {
            'query_patent': {
                'file': query_file,
                'pub_id': query_content.pub_id,
                'claim_1': query_content.claims[0] if query_content.claims else "",
                'claim_elements': [asdict(e) for e in query_content.claim_elements],
                'problem': query_content.problem[:200],
                'solution': query_content.solution[:200]
            },
            'top_results': [self._ranking_to_dict(r) for r in final_rankings],
            'analysis_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat(),
            'statistics': self.stats,
            'json_index_size': len(self.json_cache.pub_id_to_path)
        }

        # ファイル出力
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"結果保存: {output_file}")

        # コンソール出力
        self._print_results(final_rankings, query_content)

        return all_results

    def _extract_patent_content_from_json(
        self,
        patent_ref: Any,
        is_file: bool = False
    ) -> Optional[PatentContent]:
        """JSONファイルから特許内容抽出"""

        self.stats['patents_analyzed'] += 1

        try:
            if is_file:
                # ファイルパスから読み込み
                with open(patent_ref, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                file_path = patent_ref
                self.stats['json_file_reads'] += 1
            else:
                # pub_idからキャッシュ経由で取得
                data = self.json_cache.get_patent_data(patent_ref)
                if not data:
                    return None
                file_path = self.json_cache.pub_id_to_path.get(patent_ref)
                self.stats['json_cache_hits'] += 1

            # 基本情報
            bibl = data.get("bibliographic", {})
            pub_id = bibl.get("publication", {}).get("doc_number", patent_ref if not is_file else "")
            title = bibl.get("title", "")

            # IPC分類
            ipc_data = bibl.get("classification", {}).get("ipc", [])
            ipc_codes = [ipc.get("text", "") for ipc in ipc_data if ipc.get("text")]

            # 優先日
            priority_date = bibl.get("priority_claim", {}).get("date", "")

            # 請求項抽出
            claims = self._extract_claims_from_json(data)
            if not claims:
                self.logger.warning(f"請求項が見つかりません: {pub_id}")
                return None

            self.stats['claims_extracted'] += len(claims)

            # 請求項1の構成要素分解
            claim_elements = self._decompose_claim(claims[0])

            # 技術内容抽出
            desc = data.get("description", {})
            problem = self._extract_section_text(desc, "tech-problem")
            solution = self._extract_section_text(desc, "tech-solution")
            effect = self._extract_section_text(desc, "advantageous-effects")
            embodiments = self._extract_embodiments(desc)

            # 要約生成
            if problem:
                problem = self._summarize_with_llm(problem, "課題", max_length=300)
            if solution:
                solution = self._summarize_with_llm(solution, "解決手段", max_length=300)
            if effect:
                effect = self._summarize_with_llm(effect, "効果", max_length=200)

            # 埋め込み生成
            claims_text = "\n".join(claims[:3])
            claims_emb = self._generate_embedding(claims_text)
            problem_emb = self._generate_embedding(problem) if problem else None
            solution_emb = self._generate_embedding(solution) if solution else None

            return PatentContent(
                pub_id=pub_id,
                title=title,
                claims=claims,
                claim_elements=claim_elements,
                problem=problem or "記載なし",
                solution=solution or "記載なし",
                effect=effect or "記載なし",
                embodiments=embodiments,
                ipc_codes=ipc_codes,
                priority_date=priority_date,
                file_path=file_path,
                claims_embedding=claims_emb,
                problem_embedding=problem_emb,
                solution_embedding=solution_emb
            )

        except Exception as e:
            self.logger.error(f"特許内容抽出エラー: {e}", exc_info=True)
            self.stats['errors'] += 1
            return None

    def _extract_claims_from_json(self, data: Dict) -> List[str]:
        """JSONから請求項抽出"""

        claims = []

        # claimsキー
        if 'claims' in data:
            claims_data = data['claims']
            if isinstance(claims_data, list):
                for claim in claims_data:
                    if isinstance(claim, dict):
                        text = claim.get('text', '')
                        if not text and 'claim-text' in claim:
                            text = claim['claim-text']
                        if text:
                            claims.append(text)
                    elif isinstance(claim, str):
                        claims.append(claim)

        # claim-statementキー（フォールバック）
        if not claims and 'claim-statement' in data:
            claim_stmt = data['claim-statement']
            if isinstance(claim_stmt, list):
                for stmt in claim_stmt:
                    if isinstance(stmt, dict) and 'text' in stmt:
                        claims.append(stmt['text'])

        return claims

    def _assess_novelty(
        self,
        query_content: PatentContent,
        candidate_content: PatentContent
    ) -> NoveltyAssessment:
        """新規性評価（ハイブリッド版）"""

        # 1. 請求項比較
        claim_comparison = self._compare_claims(query_content, candidate_content)

        # 2. 技術的類似度
        technical_similarity = self._calculate_technical_similarity(
            query_content,
            candidate_content
        )

        # 3. グラフ分析（Neo4jから関係情報取得）
        graph_analysis = self._analyze_graph_relations(
            query_content.pub_id,
            candidate_content.pub_id
        )

        # 4. IPC類似度
        ipc_similarity = self._calculate_ipc_similarity(
            query_content.ipc_codes,
            candidate_content.ipc_codes
        )

        # 5. 課題・解決手段の一致度
        problem_solution_match = self._check_problem_solution_match(
            query_content,
            candidate_content
        )

        # 6. 実施例のサポート
        embodiment_support = self._check_embodiment_support(
            query_content.claim_elements,
            candidate_content.embodiments
        )

        # 7. 優先日の有効性
        priority_valid = self._check_priority_validity(
            query_content.priority_date,
            candidate_content.priority_date
        )

        # 8. 証拠スニペット抽出
        evidence_snippets = self._extract_evidence_snippets(
            claim_comparison,
            candidate_content
        )

        # 9. 総合評価
        blocking_power = self._calculate_blocking_power(
            claim_comparison,
            technical_similarity,
            graph_analysis.relatedness_score,
            ipc_similarity,
            problem_solution_match,
            embodiment_support
        )

        # 10. リスクレベル判定
        if blocking_power >= 0.8:
            risk_level = "HIGH"
        elif blocking_power >= 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # 11. 信頼度スコア
        confidence_score = self._calculate_confidence_score(
            claim_comparison,
            evidence_snippets
        )

        # 詳細理由生成
        detailed_reason = self._generate_detailed_reason(
            claim_comparison,
            technical_similarity,
            graph_analysis,
            problem_solution_match,
            risk_level
        )

        return NoveltyAssessment(
            pub_id=candidate_content.pub_id,
            title=candidate_content.title,
            claim_comparison=claim_comparison,
            technical_similarity=technical_similarity,
            graph_analysis=graph_analysis,
            ipc_similarity=ipc_similarity,
            problem_solution_match=problem_solution_match,
            embodiment_support=embodiment_support,
            priority_valid=priority_valid,
            novelty_risk_level=risk_level,
            blocking_power=blocking_power,
            confidence_score=confidence_score,
            detailed_reason=detailed_reason,
            evidence_snippets=evidence_snippets
        )

    def _analyze_graph_relations(
        self,
        query_pub_id: str,
        candidate_pub_id: str
    ) -> GraphAnalysis:
        """グラフDBから関係分析（pub_id形式の柔軟な対応）"""

        self.stats['graph_queries'] += 1

        # pub_idの様々な形式を試す
        def try_graph_query(q_id: str, c_id: str):
            """複数のpub_id形式で試行"""
            with self.neo4j_driver.session() as session:
                # まず、完全一致を試す
                result = session.run("""
                    OPTIONAL MATCH (p1:Patent {pub_id: $query_id})
                    OPTIONAL MATCH (p2:Patent {pub_id: $candidate_id})
                    RETURN
                        exists(p1) as q_exists,
                        exists(p2) as c_exists,
                        p1.pub_id as q_pub_id,
                        p2.pub_id as c_pub_id
                """, query_id=q_id, candidate_id=c_id).single()

                if result['q_exists'] and result['c_exists']:
                    return q_id, c_id, True

                # 部分一致を試す（数字のみ、プレフィックス付きなど）
                variations = []

                # 数字のみ抽出
                q_numbers = ''.join(re.findall(r'\d+', q_id))
                c_numbers = ''.join(re.findall(r'\d+', c_id))

                if q_numbers != q_id or c_numbers != c_id:
                    variations.append((q_numbers, c_numbers))

                # JP追加
                if not q_id.startswith('JP'):
                    variations.append((f"JP{q_id}", f"JP{c_id}"))

                # JP削除
                if q_id.startswith('JP'):
                    variations.append((q_id[2:], c_id[2:]))

                for q_var, c_var in variations:
                    result = session.run("""
                        OPTIONAL MATCH (p1:Patent {pub_id: $query_id})
                        OPTIONAL MATCH (p2:Patent {pub_id: $candidate_id})
                        RETURN exists(p1) as q_exists, exists(p2) as c_exists
                    """, query_id=q_var, candidate_id=c_var).single()

                    if result['q_exists'] and result['c_exists']:
                        self.logger.debug(f"グラフDB形式変換成功: {q_id}->{q_var}, {c_id}->{c_var}")
                        return q_var, c_var, True

                # パターンマッチング（最後の手段）
                result = session.run("""
                    OPTIONAL MATCH (p1:Patent)
                    WHERE p1.pub_id CONTAINS $query_pattern
                    OPTIONAL MATCH (p2:Patent)
                    WHERE p2.pub_id CONTAINS $candidate_pattern
                    RETURN
                        p1.pub_id as q_pub_id,
                        p2.pub_id as c_pub_id,
                        exists(p1) as q_exists,
                        exists(p2) as c_exists
                    LIMIT 1
                """, query_pattern=q_numbers if q_numbers else q_id,
                     candidate_pattern=c_numbers if c_numbers else c_id).single()

                if result and result['q_exists'] and result['c_exists']:
                    self.logger.debug(f"グラフDBパターンマッチ成功: {result['q_pub_id']}, {result['c_pub_id']}")
                    return result['q_pub_id'], result['c_pub_id'], True

                return q_id, c_id, False

        with self.neo4j_driver.session() as session:
            try:
                # 適切なpub_id形式を見つける
                actual_query_id, actual_candidate_id, found = try_graph_query(query_pub_id, candidate_pub_id)

                if not found:
                    self.logger.debug(f"グラフDBで特許が見つかりません: query={query_pub_id}, candidate={candidate_pub_id}")
                    return GraphAnalysis(
                        citation_paths=0,
                        min_citation_distance=999,
                        common_inventors=0,
                        common_ipc_exact=0,
                        common_ipc_group=0,
                        common_keywords=[],
                        relatedness_score=0.0
                    )

                self.stats['graph_hits'] += 1

                # グラフDBから技術的関係性を分析
                result = session.run("""
                    // セクション類似度分析
                    MATCH (p1:Patent {pub_id: $query_id})-[:HAS_SECTION]->(s1:Section)
                    WHERE s1.embedding IS NOT NULL
                    WITH p1, collect(s1) AS query_sections

                    MATCH (p2:Patent {pub_id: $candidate_id})-[:HAS_SECTION]->(s2:Section)
                    WHERE s2.embedding IS NOT NULL
                    WITH p1, query_sections, p2, collect(s2) AS candidate_sections

                    // トピック共通性
                    OPTIONAL MATCH (p1)-[:HAS_SECTION]->()-[:HAS_TOPIC]->(t:Topic)
                        <-[:HAS_TOPIC]-()<-[:HAS_SECTION]-(p2)
                    WITH p1, p2, query_sections, candidate_sections,
                         count(DISTINCT t) AS common_topics

                    RETURN
                        common_topics,
                        size(query_sections) AS query_section_count,
                        size(candidate_sections) AS candidate_section_count
                """, query_id=actual_query_id, candidate_id=actual_candidate_id).single()

                if result:
                    common_topics = result['common_topics'] or 0

                    # 関連性スコア計算
                    if common_topics > 0:
                        query_sections = result['query_section_count'] or 1
                        cand_sections = result['candidate_section_count'] or 1
                        # スコア計算を調整（より妥当な値になるように）
                        relatedness = min(1.0, common_topics / max(3, min(query_sections, cand_sections)))
                    else:
                        relatedness = 0.0

                    # キーワード取得（上位5個）
                    keyword_result = session.run("""
                        MATCH (p1:Patent {pub_id: $query_id})-[:HAS_SECTION]->()
                            -[:HAS_TOPIC]->(t:Topic)<-[:HAS_TOPIC]-()
                            <-[:HAS_SECTION]-(p2:Patent {pub_id: $candidate_id})
                        RETURN t.text AS keyword
                        LIMIT 5
                    """, query_id=actual_query_id, candidate_id=actual_candidate_id)

                    keywords = [r['keyword'] for r in keyword_result]

                    self.logger.debug(f"グラフ分析成功: 関連性={relatedness:.3f}, 共通トピック={common_topics}, キーワード={len(keywords)}個")

                else:
                    relatedness = 0.0
                    keywords = []
                    common_topics = 0

                return GraphAnalysis(
                    citation_paths=0,  # 今回は引用関係は省略
                    min_citation_distance=999,
                    common_inventors=0,  # 今回は発明者は省略
                    common_ipc_exact=0,  # IPCは別途計算
                    common_ipc_group=0,
                    common_keywords=keywords,
                    relatedness_score=relatedness
                )

            except Exception as e:
                self.logger.warning(f"グラフ分析エラー: {e}")
                return GraphAnalysis(
                    citation_paths=0,
                    min_citation_distance=999,
                    common_inventors=0,
                    common_ipc_exact=0,
                    common_ipc_group=0,
                    common_keywords=[],
                    relatedness_score=0.0
                )

    def _decompose_claim(self, claim_text: str) -> List[ClaimElement]:
        """請求項を構成要素に分解"""

        self.stats['llm_calls'] += 1

        prompt = f"""以下の請求項を構成要素に分解してください。
                【請求項】
                {claim_text[:2000]}

                【分解指示】
                1. 構造的要素（部品、構成）
                2. 機能的要素（動作、処理）
                3. 効果的要素（結果、作用）
                に分類して抽出してください。

                【回答形式】
                JSON形式で以下のように回答：
                {{
                    "elements": [
                        {{
                            "id": "E1",
                            "text": "要素の説明",
                            "category": "structure/function/effect",
                            "keywords": ["キーワード1", "キーワード2"]
                        }}
                    ]
                }}"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "特許請求項の構成要素分解の専門家として回答してください。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            elements = []

            for elem_data in result.get('elements', []):
                elements.append(ClaimElement(
                    element_id=elem_data.get('id', ''),
                    text=elem_data.get('text', ''),
                    category=elem_data.get('category', 'unknown'),
                    keywords=elem_data.get('keywords', [])
                ))

            return elements

        except Exception as e:
            self.logger.error(f"請求項分解エラー: {e}")
            return []

    def _compare_claims(
        self,
        query_content: PatentContent,
        candidate_content: PatentContent
    ) -> ClaimComparison:
        """請求項の詳細比較"""

        self.stats['llm_calls'] += 1

        query_claim = query_content.claims[0] if query_content.claims else ""
        candidate_claims = "\n".join(candidate_content.claims[:3])

        prompt = f"""以下の2つの特許の請求項を比較し、候補特許が入力特許の新規性を否定できるか判定してください。

            【入力特許の請求項1】
            {query_claim[:1500]}

            【入力特許の構成要素】
            {self._format_elements(query_content.claim_elements)}

            【候補特許の請求項】
            {candidate_claims[:1500]}

            【判定指示】
            1. 入力特許の各構成要素が候補特許に開示されているか
            2. 構成要素の組み合わせが開示されているか
            3. 機能的な等価性があるか

            【回答形式】
            JSON形式で回答：
            {{
                "element_mappings": [
                    {{
                        "query_element_id": "E1",
                        "is_disclosed": true/false,
                        "disclosure_type": "explicit/implicit/equivalent/not_found",
                        "evidence": "候補特許での対応箇所"
                    }}
                ],
                "all_elements_disclosed": true/false,
                "combination_disclosed": true/false,
                "functional_equivalence": true/false,
                "analysis": "詳細な分析"
            }}"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "特許の新規性判定の専門家として厳密に判定してください。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            element_mappings = []
            for elem in query_content.claim_elements:
                mapping_data = next(
                    (m for m in result.get('element_mappings', [])
                     if m.get('query_element_id') == elem.element_id),
                    None
                )

                if mapping_data:
                    element_mappings.append(ElementMapping(
                        query_element=elem,
                        candidate_element=None,
                        is_disclosed=mapping_data.get('is_disclosed', False),
                        disclosure_type=mapping_data.get('disclosure_type', 'not_found'),
                        evidence=mapping_data.get('evidence', '')
                    ))
                else:
                    element_mappings.append(ElementMapping(
                        query_element=elem,
                        candidate_element=None,
                        is_disclosed=False,
                        disclosure_type='not_found',
                        evidence=''
                    ))

            return ClaimComparison(
                all_elements_disclosed=result.get('all_elements_disclosed', False),
                element_mappings=element_mappings,
                combination_disclosed=result.get('combination_disclosed', False),
                functional_equivalence=result.get('functional_equivalence', False),
                detailed_analysis=result.get('analysis', '')
            )

        except Exception as e:
            self.logger.error(f"請求項比較エラー: {e}")
            return ClaimComparison(
                all_elements_disclosed=False,
                element_mappings=[],
                combination_disclosed=False,
                functional_equivalence=False,
                detailed_analysis="比較エラー"
            )

    def _calculate_technical_similarity(
        self,
        query: PatentContent,
        candidate: PatentContent
    ) -> float:
        """技術的類似度計算"""

        def cosine_similarity(vec1: Optional[List[float]], vec2: Optional[List[float]]) -> float:
            if not vec1 or not vec2:
                return 0.0

            if NUMPY_AVAILABLE:
                v1 = np.array(vec1)
                v2 = np.array(vec2)
                return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            else:
                dot = sum(a * b for a, b in zip(vec1, vec2))
                norm1 = sum(a * a for a in vec1) ** 0.5
                norm2 = sum(b * b for b in vec2) ** 0.5
                return dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0.0

        claim_sim = cosine_similarity(query.claims_embedding, candidate.claims_embedding)
        problem_sim = cosine_similarity(query.problem_embedding, candidate.problem_embedding)
        solution_sim = cosine_similarity(query.solution_embedding, candidate.solution_embedding)

        # 重み付け平均（請求項重視）
        return claim_sim * 0.5 + solution_sim * 0.3 + problem_sim * 0.2

    def _calculate_ipc_similarity(
        self,
        query_ipcs: List[str],
        candidate_ipcs: List[str]
    ) -> float:
        """IPC分類の類似度計算"""

        if not query_ipcs or not candidate_ipcs:
            return 0.0

        score = 0.0

        for q_ipc in query_ipcs:
            for c_ipc in candidate_ipcs:
                if q_ipc == c_ipc:
                    score += 1.0
                elif q_ipc[:4] == c_ipc[:4]:
                    score += 0.5
                elif q_ipc[:3] == c_ipc[:3]:
                    score += 0.2

        max_possible = min(len(query_ipcs), len(candidate_ipcs))
        return min(1.0, score / max_possible) if max_possible > 0 else 0.0

    def _check_problem_solution_match(
        self,
        query: PatentContent,
        candidate: PatentContent
    ) -> Dict[str, bool]:
        """課題と解決手段の一致度チェック"""

        self.stats['llm_calls'] += 1

        prompt = f"""以下の2つの特許の課題と解決手段を比較してください。

                【入力特許】
                課題: {query.problem[:300]}
                解決手段: {query.solution[:300]}

                【候補特許】
                課題: {candidate.problem[:300]}
                解決手段: {candidate.solution[:300]}

                同じ技術的問題を解決しているか判定してください。

                JSON形式で回答：
                {{
                    "same_problem": true/false,
                    "similar_solution": true/false,
                    "technical_equivalence": true/false
                }}"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "技術比較の専門家として判定してください。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            self.logger.warning(f"課題・解決手段比較エラー: {e}")
            return {
                "same_problem": False,
                "similar_solution": False,
                "technical_equivalence": False
            }

    def _check_embodiment_support(
        self,
        claim_elements: List[ClaimElement],
        embodiments: List[str]
    ) -> bool:
        """実施例による裏付けチェック"""

        if not embodiments:
            return False

        embodiment_text = "\n".join(embodiments[:2])[:2000]
        element_keywords = []

        for elem in claim_elements:
            element_keywords.extend(elem.keywords)

        found_count = 0
        for keyword in element_keywords:
            if keyword in embodiment_text:
                found_count += 1

        return found_count >= len(element_keywords) * 0.5

    def _check_priority_validity(
        self,
        query_date: Optional[str],
        candidate_date: Optional[str]
    ) -> bool:
        """優先日の有効性チェック"""

        if not query_date or not candidate_date:
            return True

        try:
            from datetime import datetime
            query_dt = datetime.fromisoformat(query_date.replace('Z', '+00:00'))
            cand_dt = datetime.fromisoformat(candidate_date.replace('Z', '+00:00'))

            return cand_dt < query_dt

        except Exception as e:
            self.logger.warning(f"日付比較エラー: {e}")
            return True

    def _extract_evidence_snippets(
        self,
        comparison: ClaimComparison,
        candidate: PatentContent
    ) -> List[str]:
        """証拠スニペット抽出"""

        snippets = []

        for mapping in comparison.element_mappings:
            if mapping.is_disclosed and mapping.evidence:
                snippets.append(f"[{mapping.disclosure_type}] {mapping.evidence[:200]}")

        if candidate.claims and comparison.all_elements_disclosed:
            snippets.append(f"[請求項1] {candidate.claims[0][:300]}")

        return snippets[:5]

    def _calculate_blocking_power(
        self,
        comparison: ClaimComparison,
        tech_sim: float,
        graph_score: float,
        ipc_sim: float,
        ps_match: Dict[str, bool],
        embodiment: bool
    ) -> float:
        """新規性否定力を計算（グラフスコア0の場合の補正付き）"""

        score = 0.0

        # 請求項の要素開示（35%）
        if comparison.all_elements_disclosed:
            score += 0.35
        else:
            disclosed_rate = sum(1 for m in comparison.element_mappings if m.is_disclosed)
            disclosed_rate /= max(1, len(comparison.element_mappings))
            score += 0.35 * disclosed_rate

        # 組み合わせの開示（15%）
        if comparison.combination_disclosed:
            score += 0.15

        # 機能的等価性（10%）
        if comparison.functional_equivalence:
            score += 0.1

        # グラフスコアが0の場合、その分を他の指標に再配分
        if graph_score > 0:
            # グラフ関連性（10%）
            score += 0.1 * graph_score
            # 技術的類似度（10%）
            score += 0.1 * tech_sim
        else:
            # グラフスコアの10%を技術的類似度とIPCに再配分
            score += 0.15 * tech_sim  # 10% + 5%追加
            score += 0.05 * ipc_sim   # 5%追加

        # IPC類似度（5%）
        score += 0.05 * ipc_sim

        # 課題・解決手段の一致（10%）
        if ps_match.get('same_problem'):
            score += 0.05
        if ps_match.get('similar_solution'):
            score += 0.05

        # 実施例サポート（5%）
        if embodiment:
            score += 0.05

        return min(1.0, score)

    def _calculate_confidence_score(
        self,
        comparison: ClaimComparison,
        evidence: List[str]
    ) -> float:
        """信頼度スコア計算"""

        score = 0.0

        explicit_count = sum(1 for m in comparison.element_mappings
                           if m.disclosure_type == 'explicit')
        if comparison.element_mappings:
            score += 0.5 * (explicit_count / len(comparison.element_mappings))

        evidence_score = min(1.0, len(evidence) / 5.0)
        score += 0.3 * evidence_score

        if comparison.detailed_analysis and len(comparison.detailed_analysis) > 100:
            score += 0.2

        return min(1.0, score)

    def _calculate_match_rate(self, assessment: NoveltyAssessment) -> float:
        """要素の一致率計算"""

        if not assessment.claim_comparison.element_mappings:
            return 0.0

        matched = sum(1 for m in assessment.claim_comparison.element_mappings
                     if m.is_disclosed)
        total = len(assessment.claim_comparison.element_mappings)

        return matched / total if total > 0 else 0.0

    def _generate_detailed_reason(
        self,
        comparison: ClaimComparison,
        tech_sim: float,
        graph: GraphAnalysis,
        ps_match: Dict[str, bool],
        risk_level: str
    ) -> str:
        """詳細な判定理由を生成"""

        reasons = []

        if comparison.all_elements_disclosed:
            reasons.append("全構成要素が開示")
        else:
            disclosed = sum(1 for m in comparison.element_mappings if m.is_disclosed)
            total = len(comparison.element_mappings)
            reasons.append(f"構成要素{disclosed}/{total}が開示")

        if comparison.combination_disclosed:
            reasons.append("組み合わせも開示")

        if comparison.functional_equivalence:
            reasons.append("機能的に等価")

        if tech_sim > 0.7:
            reasons.append(f"高い技術的類似度({tech_sim:.2f})")

        if graph.relatedness_score > 0.3:
            reasons.append(f"グラフ関連性あり({graph.relatedness_score:.2f})")
        elif graph.relatedness_score == 0:
            reasons.append("グラフ分析未実施")

        if graph.common_keywords:
            reasons.append(f"共通キーワード{len(graph.common_keywords)}個")

        if ps_match.get('same_problem'):
            reasons.append("同じ技術課題")

        risk_str = {
            "HIGH": "新規性否定の可能性が非常に高い",
            "MEDIUM": "新規性否定の可能性がある",
            "LOW": "新規性否定の可能性は低い"
        }

        return f"{risk_str[risk_level]}。{'、'.join(reasons)}。"

    # ユーティリティメソッド
    def _extract_section_text(self, desc: Dict, section_type: str) -> str:
        """セクションテキスト取得"""

        if "summary-of-invention" in desc:
            summary = desc["summary-of-invention"]
            if isinstance(summary, list) and summary:
                summary = summary[0]
            if isinstance(summary, dict) and section_type in summary:
                return self._extract_text_recursive(summary[section_type])

        if section_type in desc:
            return self._extract_text_recursive(desc[section_type])

        return ""

    def _extract_text_recursive(self, data: Any) -> str:
        """再帰的テキスト抽出"""

        if isinstance(data, str):
            return data

        texts = []

        if isinstance(data, list):
            for item in data:
                text = self._extract_text_recursive(item)
                if text:
                    texts.append(text)

        elif isinstance(data, dict):
            if "text" in data:
                return str(data["text"])

            for key, value in data.items():
                if key != "num":
                    text = self._extract_text_recursive(value)
                    if text:
                        texts.append(text)

        return "\n".join(texts)

    def _extract_embodiments(self, desc: Dict) -> List[str]:
        """実施例を抽出"""

        embodiments = []

        embodiment_keys = [
            'mode-for-invention',
            'description-of-embodiments',
            'embodiment',
            'examples'
        ]

        for key in embodiment_keys:
            if key in desc:
                text = self._extract_text_recursive(desc[key])
                if text:
                    embodiments.append(text)

        return embodiments

    def _summarize_with_llm(
        self,
        text: str,
        element_type: str,
        max_length: int = 300
    ) -> str:
        """LLMによる要約"""

        if not text or len(text.strip()) < 10:
            return f"{element_type}の記載なし"

        self.stats['llm_calls'] += 1

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"{element_type}を技術的に正確に要約してください。"},
                    {"role": "user", "content": text[:3000]}
                ],
                max_tokens=max_length,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            self.logger.warning(f"要約エラー: {e}")
            return text[:max_length]

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """埋め込み生成"""

        if not text or "記載なし" in text:
            return None

        self.stats['embeddings_generated'] += 1

        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=[text]
            )
            return response.data[0].embedding

        except Exception as e:
            self.logger.warning(f"埋め込みエラー: {e}")
            return None

    def _format_elements(self, elements: List[ClaimElement]) -> str:
        """構成要素をフォーマット"""

        lines = []
        for elem in elements:
            lines.append(f"{elem.element_id}: {elem.text} ({elem.category})")
            if elem.keywords:
                lines.append(f"  キーワード: {', '.join(elem.keywords)}")

        return "\n".join(lines)

    def _log_patent_content(self, label: str, content: PatentContent):
        """特許内容のログ出力"""

        self.logger.info(f"\n{label}:")
        self.logger.info(f"  公開番号: {content.pub_id}")
        self.logger.info(f"  タイトル: {content.title[:80]}...")
        self.logger.info(f"  請求項数: {len(content.claims)}")

        if content.claims:
            self.logger.info(f"  請求項1: {content.claims[0][:150]}...")

        self.logger.info(f"  構成要素数: {len(content.claim_elements)}")

        for elem in content.claim_elements[:3]:
            self.logger.info(f"    - {elem.element_id}: {elem.text[:50]}... ({elem.category})")

        self.logger.info(f"  IPC: {', '.join(content.ipc_codes[:5])}")

        if content.file_path:
            self.logger.info(f"  ソース: {content.file_path}")

    def _ranking_to_dict(self, ranking: FinalRanking) -> Dict:
        """ランキング結果を辞書に変換"""

        assessment = ranking.assessment

        return {
            'rank': ranking.rank,
            'pub_id': ranking.pub_id,
            'title': ranking.title,
            'ipc_codes': ranking.ipc_codes,
            'scores': {
                'blocking_power': round(ranking.blocking_power, 3),
                'claim_match_rate': round(ranking.claim_match_rate, 3),
                'graph_score': round(ranking.graph_score, 3),
                'technical_similarity': round(assessment.technical_similarity, 3),
                'confidence': round(assessment.confidence_score, 3)
            },
            'claim_analysis': {
                'all_elements_disclosed': assessment.claim_comparison.all_elements_disclosed,
                'combination_disclosed': assessment.claim_comparison.combination_disclosed,
                'functional_equivalence': assessment.claim_comparison.functional_equivalence,
                'element_details': [
                    {
                        'element': m.query_element.text[:100],
                        'disclosed': m.is_disclosed,
                        'type': m.disclosure_type
                    }
                    for m in assessment.claim_comparison.element_mappings[:5]
                ]
            },
            'graph_analysis': {
                'relatedness_score': assessment.graph_analysis.relatedness_score,
                'common_keywords': assessment.graph_analysis.common_keywords[:5]
            },
            'risk_level': assessment.novelty_risk_level,
            'detailed_reason': assessment.detailed_reason,
            'evidence_snippets': assessment.evidence_snippets[:3],
            'original_rank': ranking.original_rank,
            'rank_change': ranking.rank_change
        }

    def _print_results(self, rankings: List[FinalRanking], query: PatentContent):
        """結果の表示"""

        print("\n" + "="*80)
        print("【ハイブリッド新規性評価結果】")
        print("="*80)

        print("\n【入力特許】")
        print(f"公開番号: {query.pub_id}")
        print(f"請求項1: {query.claims[0][:200]}...")
        print(f"\n構成要素:")
        for elem in query.claim_elements[:5]:
            print(f"  • {elem.element_id}: {elem.text[:80]}...")

        print("\n" + "-"*80)
        print("【新規性否定可能性ランキング TOP 10】")
        print("-"*80)

        for r in rankings:
            assessment = r.assessment

            risk_symbols = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
            risk_symbol = risk_symbols.get(assessment.novelty_risk_level, "⚪")

            print(f"\n{r.rank}. {risk_symbol} {r.pub_id}")
            print(f"   タイトル: {r.title[:70]}...")
            print(f"   新規性否定力: {r.blocking_power:.1%}")
            print(f"   請求項一致率: {r.claim_match_rate:.1%}")

            disclosed_elems = sum(1 for m in assessment.claim_comparison.element_mappings
                                if m.is_disclosed)
            total_elems = len(assessment.claim_comparison.element_mappings)
            print(f"   構成要素: {disclosed_elems}/{total_elems}個が開示")

            highlights = []
            if assessment.claim_comparison.all_elements_disclosed:
                highlights.append("全要素開示")
            if assessment.claim_comparison.combination_disclosed:
                highlights.append("組合せ開示")
            if assessment.claim_comparison.functional_equivalence:
                highlights.append("機能等価")
            if assessment.graph_analysis.relatedness_score > 0.3:
                highlights.append(f"グラフ関連{assessment.graph_analysis.relatedness_score:.1%}")
            elif assessment.graph_analysis.relatedness_score == 0:
                highlights.append("グラフ未分析")

            if highlights:
                print(f"   特記: {', '.join(highlights)}")

            if assessment.graph_analysis.common_keywords:
                print(f"   共通キーワード: {', '.join(assessment.graph_analysis.common_keywords[:3])}")

            print(f"   判定: {assessment.detailed_reason[:150]}...")

            if assessment.evidence_snippets:
                print(f"   証拠: {assessment.evidence_snippets[0][:100]}...")

            if r.rank_change > 0:
                print(f"   順位変動: ↑{r.rank_change}")
            elif r.rank_change < 0:
                print(f"   順位変動: ↓{abs(r.rank_change)}")

        # グラフDB統計情報
        print("\n" + "-"*80)
        print(f"【グラフDB分析統計】")
        print(f"  クエリ数: {self.stats['graph_queries']}")
        print(f"  ヒット数: {self.stats['graph_hits']}")
        print(f"  ヒット率: {self.stats['graph_hits']/max(1, self.stats['graph_queries'])*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(
        description="ハイブリッド新規性評価システム（JSON+Graph）"
    )
    parser.add_argument("--stage2-json", required=True, help="Stage2結果JSON")
    parser.add_argument("--query-file", required=True, help="クエリ特許ファイル")
    parser.add_argument("--json-dir", default=JSON_DATA_DIR, help="特許JSONファイルディレクトリ")
    parser.add_argument("--output", help="出力ファイル名")
    parser.add_argument("--top-k", type=int, default=10, help="上位K件を出力")
    parser.add_argument("--log-file", help="ログファイル")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    setup_logging(log_file=args.log_file, verbose=args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("システム起動")

    engine = HybridNoveltyEngine(json_dir=args.json_dir)

    try:
        results = engine.run_analysis(
            stage2_json=args.stage2_json,
            query_file=args.query_file,
            output_file=args.output or "hybrid_novelty_analysis.json",
            top_k=args.top_k
        )

        logger.info("分析完了")
        return 0

    except Exception as e:
        logger.error(f"実行エラー: {e}", exc_info=True)
        return 1
    finally:
        engine.close()

if __name__ == "__main__":
    sys.exit(main())
