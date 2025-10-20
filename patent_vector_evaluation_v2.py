"""
特許ベクトル検索の評価スクリプト（改良版）
Cosmos DBとElasticsearchのデータ構造を正しく理解した上での評価
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from azure.cosmos import CosmosClient
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import AzureOpenAI

# 環境変数の読み込み
load_dotenv()

VECTOR_FIELD = os.getenv("ES_VECTOR_FIELD", "claims_vector")


class PatentVectorEvaluator:
    def __init__(self):
        """初期化"""
        # Elasticsearch接続
        self.es = Elasticsearch('http://localhost:9200')

        # Azure OpenAI接続
        self.openai_client = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            api_version="2024-02-01"
        )
        self.embedding_model = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'text-embedding-3-large')

        # Cosmos DB接続
        self.cosmos_client = CosmosClient(
            os.getenv('COSMOS_ENDPOINT'),
            os.getenv('COSMOS_KEY')
        )
        database = self.cosmos_client.get_database_client(
            os.getenv('COSMOS_DATABASE', 'patent_json')
        )
        self.cosmos_container = database.get_container_client(
            os.getenv('COSMOS_CONTAINER', 'patent_csv1')
        )

        # インデックス名
        self.index_name = 'patent_vectors'

    def get_embedding(self, text: str) -> List[float]:
        """テキストのembeddingを取得"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    def search_by_syutugan_cosmos(self, syutugan_id: str) -> List[str]:
        """
        Cosmos DBでsyutuganフィールドに特定のIDを含むドキュメントを検索
        これは「syutugan_idを引用している特許」を見つける
        """
        query = f"""
        SELECT c.id, c.metadata.reference.syutugan
        FROM c
        WHERE ARRAY_CONTAINS(c.metadata.reference.syutugan, '{syutugan_id}')
        """

        try:
            items = list(self.cosmos_container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))

            # IDから特許番号を抽出（例: "2015163142_20150910" -> "JP2015163142A"）
            patent_ids = []
            for item in items:
                doc_id = item['id']
                # IDの最初の部分が特許番号
                patent_num = doc_id.split('_')[0]
                if patent_num != syutugan_id.replace('JP', '').replace('A', ''):
                    patent_ids.append(f"JP{patent_num}A")

            return patent_ids
        except Exception as e:
            print(f"Error searching Cosmos DB: {e}")
            return []

    def search_by_syutugan_elasticsearch(self, syutugan_id: str, top_k: int = 10) -> List[Dict]:
        """
        Elasticsearchでsyutuganフィールドに特定のIDを含むドキュメントを検索
        """
        query = {
            "size": top_k,
            "query": {
                "term": {
                    "reference.syutugan": syutugan_id
                }
            }
        }

        try:
            response = self.es.search(index=self.index_name, body=query)
            results = []
            for hit in response['hits']['hits']:
                doc = hit['_source']
                # IDから特許番号を抽出
                patent_num = hit['_id']
                if patent_num != syutugan_id.replace('JP', '').replace('A', ''):
                    results.append({
                        'patent_id': f"JP{patent_num}A",
                        'score': hit['_score'],
                        'title': doc.get('title', ''),
                        'syutugan': doc.get('reference', {}).get('syutugan', [])
                    })
            return results
        except Exception as e:
            print(f"Error searching Elasticsearch: {e}")
            return []

    def vector_search_elasticsearch(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """Elasticsearchでベクトル検索を実行"""
        # クエリのembeddingを取得
        query_embedding = self.get_embedding(query_text)
        if not query_embedding:
            return []

        # ベクトル検索クエリ
        query = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, '{VECTOR_FIELD}') + 1.0",
                        "params": {
                            "query_vector": query_embedding
                        }
                    }
                }
            }
        }

        try:
            response = self.es.search(index=self.index_name, body=query)
            results = []
            for hit in response['hits']['hits']:
                doc = hit['_source']
                results.append({
                    'patent_id': f"JP{hit['_id']}A",
                    'score': hit['_score'],
                    'title': doc.get('title', ''),
                    'summary': doc.get('summary', '')[:200],
                    'syutugan': doc.get('reference', {}).get('syutugan', [])
                })
            return results
        except Exception as e:
            print(f"Error in vector search: {e}")
            return []

    def hybrid_search_elasticsearch(self, query_text: str, syutugan_id: str, top_k: int = 10) -> List[Dict]:
        """Elasticsearchでハイブリッド検索（Term + Vector）を実行"""
        # クエリのembeddingを取得
        query_embedding = self.get_embedding(query_text)
        if not query_embedding:
            return self.search_by_syutugan_elasticsearch(syutugan_id, top_k)

        # ハイブリッド検索クエリ（Term検索とVector検索の組み合わせ）
        query = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "term": {
                                "reference.syutugan": {
                                    "value": syutugan_id,
                                    "boost": 10.0  # Term検索に高いブースト
                                }
                            }
                        },
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": f"cosineSimilarity(params.query_vector, '{VECTOR_FIELD}') + 1.0",
                                    "params": {
                                        "query_vector": query_embedding
                                    }
                                },
                                "boost": 1.0  # Vector検索のブースト
                            }
                        }
                    ]
                }
            }
        }

        try:
            response = self.es.search(index=self.index_name, body=query)
            results = []
            for hit in response['hits']['hits']:
                doc = hit['_source']
                results.append({
                    'patent_id': f"JP{hit['_id']}A",
                    'score': hit['_score'],
                    'title': doc.get('title', ''),
                    'summary': doc.get('summary', '')[:200],
                    'syutugan': doc.get('reference', {}).get('syutugan', [])
                })
            return results
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return []

    def evaluate_test_cases_extended(self, test_file: str):
        """拡張テストケース評価（top-10/50/100 + ハイブリッド検索）"""
        # テストケースを読み込み
        df = pd.read_csv(test_file)
        print(f"Loaded {len(df)} test cases")

        results = []

        for idx, row in df.iterrows():
            case_id = row['case_id']
            syutugan = row['syutugan']
            expected = row['ax_docs']

            print(f"\nEvaluating {case_id}: syutugan={syutugan}, expected={expected}")

            # 1. Cosmos DBでの検索（正解確認）
            cosmos_results = self.search_by_syutugan_cosmos(syutugan)
            cosmos_found = expected in cosmos_results
            print(f"  Cosmos DB: Found={cosmos_found}, Results={cosmos_results[:3]}")

            # 2. Elasticsearchでの検索
            es_results = self.search_by_syutugan_elasticsearch(syutugan, top_k=100)
            es_patent_ids = [r['patent_id'] for r in es_results]
            es_found = expected in es_patent_ids
            es_rank = es_patent_ids.index(expected) + 1 if es_found else -1
            print(f"  Elasticsearch: Found={es_found}, Rank={es_rank}, Top3={es_patent_ids[:3]}")

            # 3. syutuganの特許の要約を取得
            query_text = ""
            try:
                patent_num = syutugan.replace('JP', '').replace('A', '')
                syutugan_query = f"SELECT c.summary FROM c WHERE c.id LIKE '{patent_num}%'"
                syutugan_items = list(self.cosmos_container.query_items(
                    query=syutugan_query,
                    enable_cross_partition_query=True
                ))
                if syutugan_items:
                    query_text = syutugan_items[0].get('summary', '')[:500]
            except Exception:
                pass

            # 4. ベクトル検索（複数のtop_k値で評価）
            vector_results_10 = []
            vector_results_50 = []
            vector_results_100 = []

            vector_found_10 = vector_found_50 = vector_found_100 = False
            vector_rank_10 = vector_rank_50 = vector_rank_100 = -1

            if query_text:
                # Top-10
                vector_results_10 = self.vector_search_elasticsearch(query_text, top_k=10)
                vector_patent_ids_10 = [r['patent_id'] for r in vector_results_10]
                vector_found_10 = expected in vector_patent_ids_10
                vector_rank_10 = vector_patent_ids_10.index(expected) + 1 if vector_found_10 else -1

                # Top-50
                vector_results_50 = self.vector_search_elasticsearch(query_text, top_k=50)
                vector_patent_ids_50 = [r['patent_id'] for r in vector_results_50]
                vector_found_50 = expected in vector_patent_ids_50
                vector_rank_50 = vector_patent_ids_50.index(expected) + 1 if vector_found_50 else -1

                # Top-100
                vector_results_100 = self.vector_search_elasticsearch(query_text, top_k=100)
                vector_patent_ids_100 = [r['patent_id'] for r in vector_results_100]
                vector_found_100 = expected in vector_patent_ids_100
                vector_rank_100 = vector_patent_ids_100.index(expected) + 1 if vector_found_100 else -1

                print(f"  Vector Search Top-10: Found={vector_found_10}, Rank={vector_rank_10}")
                print(f"  Vector Search Top-50: Found={vector_found_50}, Rank={vector_rank_50}")
                print(f"  Vector Search Top-100: Found={vector_found_100}, Rank={vector_rank_100}")

            # 5. ハイブリッド検索（複数のtop_k値で評価）
            hybrid_results_10 = []
            hybrid_results_50 = []
            hybrid_results_100 = []

            hybrid_found_10 = hybrid_found_50 = hybrid_found_100 = False
            hybrid_rank_10 = hybrid_rank_50 = hybrid_rank_100 = -1

            if query_text:
                # Top-10
                hybrid_results_10 = self.hybrid_search_elasticsearch(query_text, syutugan, top_k=10)
                hybrid_patent_ids_10 = [r['patent_id'] for r in hybrid_results_10]
                hybrid_found_10 = expected in hybrid_patent_ids_10
                hybrid_rank_10 = hybrid_patent_ids_10.index(expected) + 1 if hybrid_found_10 else -1

                # Top-50
                hybrid_results_50 = self.hybrid_search_elasticsearch(query_text, syutugan, top_k=50)
                hybrid_patent_ids_50 = [r['patent_id'] for r in hybrid_results_50]
                hybrid_found_50 = expected in hybrid_patent_ids_50
                hybrid_rank_50 = hybrid_patent_ids_50.index(expected) + 1 if hybrid_found_50 else -1

                # Top-100
                hybrid_results_100 = self.hybrid_search_elasticsearch(query_text, syutugan, top_k=100)
                hybrid_patent_ids_100 = [r['patent_id'] for r in hybrid_results_100]
                hybrid_found_100 = expected in hybrid_patent_ids_100
                hybrid_rank_100 = hybrid_patent_ids_100.index(expected) + 1 if hybrid_found_100 else -1

                print(f"  Hybrid Search Top-10: Found={hybrid_found_10}, Rank={hybrid_rank_10}")
                print(f"  Hybrid Search Top-50: Found={hybrid_found_50}, Rank={hybrid_rank_50}")
                print(f"  Hybrid Search Top-100: Found={hybrid_found_100}, Rank={hybrid_rank_100}")

            results.append({
                'case_id': case_id,
                'syutugan': syutugan,
                'expected': expected,
                'cosmos_found': cosmos_found,
                'cosmos_results': cosmos_results[:5],
                'es_found': es_found,
                'es_rank': es_rank,
                'es_top5': es_patent_ids[:5],
                # Vector Search結果
                'vector_found_10': vector_found_10,
                'vector_rank_10': vector_rank_10,
                'vector_found_50': vector_found_50,
                'vector_rank_50': vector_rank_50,
                'vector_found_100': vector_found_100,
                'vector_rank_100': vector_rank_100,
                'vector_top5_10': [r['patent_id'] for r in vector_results_10[:5]],
                # Hybrid Search結果
                'hybrid_found_10': hybrid_found_10,
                'hybrid_rank_10': hybrid_rank_10,
                'hybrid_found_50': hybrid_found_50,
                'hybrid_rank_50': hybrid_rank_50,
                'hybrid_found_100': hybrid_found_100,
                'hybrid_rank_100': hybrid_rank_100,
                'hybrid_top5_10': [r['patent_id'] for r in hybrid_results_10[:5]]
            })

            # 進捗表示
            if (idx + 1) % 10 == 0:
                print(f"\nProgress: {idx + 1}/{len(df)}")
                self.print_intermediate_stats_extended(results)

            # レート制限対策
            time.sleep(0.5)

        # 結果を保存
        self.save_results_extended(results)

        # 統計を表示
        self.print_final_stats_extended(results)

        return results

    def evaluate_test_cases(self, test_file: str):
        """テストケースを評価"""
        # テストケースを読み込み
        df = pd.read_csv(test_file)
        print(f"Loaded {len(df)} test cases")

        results = []

        for idx, row in df.iterrows():
            case_id = row['case_id']
            syutugan = row['syutugan']
            expected = row['ax_docs']

            print(f"\nEvaluating {case_id}: syutugan={syutugan}, expected={expected}")

            # 1. Cosmos DBでの検索（正解確認）
            cosmos_results = self.search_by_syutugan_cosmos(syutugan)
            cosmos_found = expected in cosmos_results
            print(f"  Cosmos DB: Found={cosmos_found}, Results={cosmos_results[:3]}")

            # 2. Elasticsearchでの検索
            es_results = self.search_by_syutugan_elasticsearch(syutugan, top_k=10)
            es_patent_ids = [r['patent_id'] for r in es_results]
            es_found = expected in es_patent_ids
            es_rank = es_patent_ids.index(expected) + 1 if es_found else -1
            print(f"  Elasticsearch: Found={es_found}, Rank={es_rank}, Top3={es_patent_ids[:3]}")

            # 3. ベクトル検索（syutuganの特許の要約を取得してクエリとする）
            # まずsyutuganの特許の情報を取得
            query_text = ""
            try:
                patent_num = syutugan.replace('JP', '').replace('A', '')
                syutugan_query = f"SELECT c.summary FROM c WHERE c.id LIKE '{patent_num}%'"
                syutugan_items = list(self.cosmos_container.query_items(
                    query=syutugan_query,
                    enable_cross_partition_query=True
                ))
                if syutugan_items:
                    query_text = syutugan_items[0].get('summary', '')[:500]
            except Exception:
                pass

            vector_results = []
            vector_found = False
            vector_rank = -1

            if query_text:
                vector_results = self.vector_search_elasticsearch(query_text, top_k=10)
                vector_patent_ids = [r['patent_id'] for r in vector_results]
                vector_found = expected in vector_patent_ids
                vector_rank = vector_patent_ids.index(expected) + 1 if vector_found else -1
                print(f"  Vector Search: Found={vector_found}, Rank={vector_rank}, Top3={vector_patent_ids[:3]}")

            results.append({
                'case_id': case_id,
                'syutugan': syutugan,
                'expected': expected,
                'cosmos_found': cosmos_found,
                'cosmos_results': cosmos_results[:5],
                'es_found': es_found,
                'es_rank': es_rank,
                'es_top5': es_patent_ids[:5],
                'vector_found': vector_found,
                'vector_rank': vector_rank,
                'vector_top5': [r['patent_id'] for r in vector_results[:5]]
            })

            # 進捗表示
            if (idx + 1) % 10 == 0:
                print(f"\nProgress: {idx + 1}/{len(df)}")
                self.print_intermediate_stats(results)

            # レート制限対策
            time.sleep(0.5)

        # 結果を保存
        self.save_results(results)

        # 統計を表示
        self.print_final_stats(results)

        return results

    def print_intermediate_stats(self, results):
        """中間統計を表示"""
        total = len(results)
        cosmos_accuracy = sum(1 for r in results if r['cosmos_found']) / total
        es_accuracy = sum(1 for r in results if r['es_found']) / total
        vector_accuracy = sum(1 for r in results if r['vector_found']) / total

        print(f"  Intermediate Stats ({total} cases):")
        print(f"    Cosmos DB Accuracy: {cosmos_accuracy:.2%}")
        print(f"    Elasticsearch Accuracy: {es_accuracy:.2%}")
        print(f"    Vector Search Accuracy: {vector_accuracy:.2%}")

    def print_final_stats(self, results):
        """最終統計を表示"""
        total = len(results)

        print("\n" + "="*80)
        print("FINAL EVALUATION RESULTS")
        print("="*80)

        # Cosmos DB統計
        cosmos_found = sum(1 for r in results if r['cosmos_found'])
        print("\nCosmos DB (Ground Truth):")
        print(f"  Found: {cosmos_found}/{total} ({cosmos_found/total:.2%})")

        # Elasticsearch統計
        es_found = sum(1 for r in results if r['es_found'])
        es_top3 = sum(1 for r in results if 0 < r['es_rank'] <= 3)
        es_top5 = sum(1 for r in results if 0 < r['es_rank'] <= 5)
        print("\nElasticsearch (Term Search):")
        print(f"  Found: {es_found}/{total} ({es_found/total:.2%})")
        print(f"  Top-3: {es_top3}/{total} ({es_top3/total:.2%})")
        print(f"  Top-5: {es_top5}/{total} ({es_top5/total:.2%})")

        # Vector Search統計
        vector_found = sum(1 for r in results if r['vector_found'])
        vector_top3 = sum(1 for r in results if 0 < r['vector_rank'] <= 3)
        vector_top5 = sum(1 for r in results if 0 < r['vector_rank'] <= 5)
        print("\nVector Search:")
        print(f"  Found: {vector_found}/{total} ({vector_found/total:.2%})")
        print(f"  Top-3: {vector_top3}/{total} ({vector_top3/total:.2%})")
        print(f"  Top-5: {vector_top5}/{total} ({vector_top5/total:.2%})")

        # 平均ランク（見つかった場合のみ）
        es_ranks = [r['es_rank'] for r in results if r['es_rank'] > 0]
        vector_ranks = [r['vector_rank'] for r in results if r['vector_rank'] > 0]

        if es_ranks:
            print("\nAverage Rank (when found):")
            print(f"  Elasticsearch: {np.mean(es_ranks):.2f}")
        if vector_ranks:
            print(f"  Vector Search: {np.mean(vector_ranks):.2f}")

    def print_intermediate_stats_extended(self, results):
        """拡張中間統計を表示"""
        total = len(results)
        cosmos_accuracy = sum(1 for r in results if r['cosmos_found']) / total
        es_accuracy = sum(1 for r in results if r['es_found']) / total

        vector_accuracy_10 = sum(1 for r in results if r['vector_found_10']) / total
        vector_accuracy_50 = sum(1 for r in results if r['vector_found_50']) / total
        vector_accuracy_100 = sum(1 for r in results if r['vector_found_100']) / total

        hybrid_accuracy_10 = sum(1 for r in results if r['hybrid_found_10']) / total
        hybrid_accuracy_50 = sum(1 for r in results if r['hybrid_found_50']) / total
        hybrid_accuracy_100 = sum(1 for r in results if r['hybrid_found_100']) / total

        print(f"  Intermediate Stats ({total} cases):")
        print(f"    Cosmos DB Accuracy: {cosmos_accuracy:.2%}")
        print(f"    Elasticsearch Accuracy: {es_accuracy:.2%}")
        print(f"    Vector Search Top-10: {vector_accuracy_10:.2%}")
        print(f"    Vector Search Top-50: {vector_accuracy_50:.2%}")
        print(f"    Vector Search Top-100: {vector_accuracy_100:.2%}")
        print(f"    Hybrid Search Top-10: {hybrid_accuracy_10:.2%}")
        print(f"    Hybrid Search Top-50: {hybrid_accuracy_50:.2%}")
        print(f"    Hybrid Search Top-100: {hybrid_accuracy_100:.2%}")

    def print_final_stats_extended(self, results):
        """拡張最終統計を表示"""
        total = len(results)

        print("\n" + "="*80)
        print("EXTENDED EVALUATION RESULTS")
        print("="*80)

        # Cosmos DB統計
        cosmos_found = sum(1 for r in results if r['cosmos_found'])
        print("\nCosmos DB (Ground Truth):")
        print(f"  Found: {cosmos_found}/{total} ({cosmos_found/total:.2%})")

        # Elasticsearch統計
        es_found = sum(1 for r in results if r['es_found'])
        print("\nElasticsearch (Term Search):")
        print(f"  Found: {es_found}/{total} ({es_found/total:.2%})")

        # Vector Search統計
        vector_found_10 = sum(1 for r in results if r['vector_found_10'])
        vector_found_50 = sum(1 for r in results if r['vector_found_50'])
        vector_found_100 = sum(1 for r in results if r['vector_found_100'])

        vector_top3_10 = sum(1 for r in results if 0 < r['vector_rank_10'] <= 3)
        vector_top5_10 = sum(1 for r in results if 0 < r['vector_rank_10'] <= 5)
        vector_top3_50 = sum(1 for r in results if 0 < r['vector_rank_50'] <= 3)
        vector_top5_50 = sum(1 for r in results if 0 < r['vector_rank_50'] <= 5)
        vector_top3_100 = sum(1 for r in results if 0 < r['vector_rank_100'] <= 3)
        vector_top5_100 = sum(1 for r in results if 0 < r['vector_rank_100'] <= 5)

        print("\nVector Search:")
        print(f"  Top-10 Found: {vector_found_10}/{total} ({vector_found_10/total:.2%})")
        print(f"  Top-10 Top-3: {vector_top3_10}/{total} ({vector_top3_10/total:.2%})")
        print(f"  Top-10 Top-5: {vector_top5_10}/{total} ({vector_top5_10/total:.2%})")
        print(f"  Top-50 Found: {vector_found_50}/{total} ({vector_found_50/total:.2%})")
        print(f"  Top-50 Top-3: {vector_top3_50}/{total} ({vector_top3_50/total:.2%})")
        print(f"  Top-50 Top-5: {vector_top5_50}/{total} ({vector_top5_50/total:.2%})")
        print(f"  Top-100 Found: {vector_found_100}/{total} ({vector_found_100/total:.2%})")
        print(f"  Top-100 Top-3: {vector_top3_100}/{total} ({vector_top3_100/total:.2%})")
        print(f"  Top-100 Top-5: {vector_top5_100}/{total} ({vector_top5_100/total:.2%})")

        # Hybrid Search統計
        hybrid_found_10 = sum(1 for r in results if r['hybrid_found_10'])
        hybrid_found_50 = sum(1 for r in results if r['hybrid_found_50'])
        hybrid_found_100 = sum(1 for r in results if r['hybrid_found_100'])

        hybrid_top3_10 = sum(1 for r in results if 0 < r['hybrid_rank_10'] <= 3)
        hybrid_top5_10 = sum(1 for r in results if 0 < r['hybrid_rank_10'] <= 5)
        hybrid_top3_50 = sum(1 for r in results if 0 < r['hybrid_rank_50'] <= 3)
        hybrid_top5_50 = sum(1 for r in results if 0 < r['hybrid_rank_50'] <= 5)
        hybrid_top3_100 = sum(1 for r in results if 0 < r['hybrid_rank_100'] <= 3)
        hybrid_top5_100 = sum(1 for r in results if 0 < r['hybrid_rank_100'] <= 5)

        print("\nHybrid Search:")
        print(f"  Top-10 Found: {hybrid_found_10}/{total} ({hybrid_found_10/total:.2%})")
        print(f"  Top-10 Top-3: {hybrid_top3_10}/{total} ({hybrid_top3_10/total:.2%})")
        print(f"  Top-10 Top-5: {hybrid_top5_10}/{total} ({hybrid_top5_10/total:.2%})")
        print(f"  Top-50 Found: {hybrid_found_50}/{total} ({hybrid_found_50/total:.2%})")
        print(f"  Top-50 Top-3: {hybrid_top3_50}/{total} ({hybrid_top3_50/total:.2%})")
        print(f"  Top-50 Top-5: {hybrid_top5_50}/{total} ({hybrid_top5_50/total:.2%})")
        print(f"  Top-100 Found: {hybrid_found_100}/{total} ({hybrid_found_100/total:.2%})")
        print(f"  Top-100 Top-3: {hybrid_top3_100}/{total} ({hybrid_top3_100/total:.2%})")
        print(f"  Top-100 Top-5: {hybrid_top5_100}/{total} ({hybrid_top5_100/total:.2%})")

        # 平均ランク（見つかった場合のみ）
        vector_ranks_10 = [r['vector_rank_10'] for r in results if r['vector_rank_10'] > 0]
        vector_ranks_50 = [r['vector_rank_50'] for r in results if r['vector_rank_50'] > 0]
        vector_ranks_100 = [r['vector_rank_100'] for r in results if r['vector_rank_100'] > 0]

        hybrid_ranks_10 = [r['hybrid_rank_10'] for r in results if r['hybrid_rank_10'] > 0]
        hybrid_ranks_50 = [r['hybrid_rank_50'] for r in results if r['hybrid_rank_50'] > 0]
        hybrid_ranks_100 = [r['hybrid_rank_100'] for r in results if r['hybrid_rank_100'] > 0]

        print("\nAverage Rank (when found):")
        if vector_ranks_10:
            print(f"  Vector Search Top-10: {np.mean(vector_ranks_10):.2f}")
        if vector_ranks_50:
            print(f"  Vector Search Top-50: {np.mean(vector_ranks_50):.2f}")
        if vector_ranks_100:
            print(f"  Vector Search Top-100: {np.mean(vector_ranks_100):.2f}")
        if hybrid_ranks_10:
            print(f"  Hybrid Search Top-10: {np.mean(hybrid_ranks_10):.2f}")
        if hybrid_ranks_50:
            print(f"  Hybrid Search Top-50: {np.mean(hybrid_ranks_50):.2f}")
        if hybrid_ranks_100:
            print(f"  Hybrid Search Top-100: {np.mean(hybrid_ranks_100):.2f}")

    def save_results_extended(self, results):
        """拡張結果をファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 詳細結果をJSON形式で保存
        with open(f'evaluation_results_extended_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # サマリーをCSV形式で保存
        df_results = pd.DataFrame(results)
        df_results.to_csv(f'evaluation_summary_extended_{timestamp}.csv', index=False)

        print("\nExtended results saved to:")
        print(f"  - evaluation_results_extended_{timestamp}.json")
        print(f"  - evaluation_summary_extended_{timestamp}.csv")

    def save_results(self, results):
        """結果をファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 詳細結果をJSON形式で保存
        with open(f'evaluation_results_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # サマリーをCSV形式で保存
        df_results = pd.DataFrame(results)
        df_results.to_csv(f'evaluation_summary_{timestamp}.csv', index=False)

        print("\nResults saved to:")
        print(f"  - evaluation_results_{timestamp}.json")
        print(f"  - evaluation_summary_{timestamp}.csv")


def main():
    """メイン処理"""
    evaluator = PatentVectorEvaluator()

    # テストケースファイル
    test_file = 'filtered_test_cases_20250813_174350.csv'

    print(f"Starting extended evaluation with {test_file}")
    print("="*80)

    # 拡張評価実行（top-10/50/100 + ハイブリッド検索）
    evaluator.evaluate_test_cases_extended(test_file)

    print("\nExtended evaluation completed!")


if __name__ == "__main__":
    main()
