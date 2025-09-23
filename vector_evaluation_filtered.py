"""
両方とも存在しているケースだけでベクトル検索の評価を行うスクリプト
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from azure.cosmos import CosmosClient
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from openai import AzureOpenAI

# 環境変数の読み込み
load_dotenv()


class FilteredVectorEvaluator:
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

    def get_embedding(self, text: str):
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

    def vector_search_elasticsearch(self, query_text: str, top_k: int = 10):
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
                        "source": "cosineSimilarity(params.query_vector, 'summary_vector') + 1.0",
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
                    'summary': doc.get('summary', '')[:200]
                })
            return results
        except Exception as e:
            print(f"Error in vector search: {e}")
            return []

    def evaluate_filtered_cases(self):
        """両方とも存在するケースのみでベクトル検索を評価"""

        # 存在確認結果を読み込み
        existence_df = pd.read_csv('elasticsearch_existence_check.csv')

        # 両方とも存在するケースのみをフィルタ
        filtered_df = existence_df[existence_df['both_exist'] == True].copy()

        print(f"Total test cases: {len(existence_df)}")
        print(f"Filtered cases (both exist): {len(filtered_df)}")
        print(f"Filtering ratio: {len(filtered_df)/len(existence_df):.2%}")

        results = []

        for idx, row in filtered_df.iterrows():
            case_id = row['case_id']
            syutugan = row['syutugan']
            expected = row['ax_docs']

            print(f"\nEvaluating {case_id}: syutugan={syutugan}, expected={expected}")

            # syutuganの特許の要約を取得
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
            except Exception as e:
                print(f"  Error getting summary: {e}")
                continue

            if not query_text:
                print(f"  No summary found for {syutugan}")
                continue

            # ベクトル検索（複数のtop_k値で評価）
            vector_results_10 = self.vector_search_elasticsearch(query_text, top_k=10)
            vector_results_50 = self.vector_search_elasticsearch(query_text, top_k=50)
            vector_results_100 = self.vector_search_elasticsearch(query_text, top_k=100)

            # 結果の評価
            vector_patent_ids_10 = [r['patent_id'] for r in vector_results_10]
            vector_patent_ids_50 = [r['patent_id'] for r in vector_results_50]
            vector_patent_ids_100 = [r['patent_id'] for r in vector_results_100]

            vector_found_10 = expected in vector_patent_ids_10
            vector_found_50 = expected in vector_patent_ids_50
            vector_found_100 = expected in vector_patent_ids_100

            vector_rank_10 = vector_patent_ids_10.index(expected) + 1 if vector_found_10 else -1
            vector_rank_50 = vector_patent_ids_50.index(expected) + 1 if vector_found_50 else -1
            vector_rank_100 = vector_patent_ids_100.index(expected) + 1 if vector_found_100 else -1

            print(f"  Vector Search Top-10: Found={vector_found_10}, Rank={vector_rank_10}")
            print(f"  Vector Search Top-50: Found={vector_found_50}, Rank={vector_rank_50}")
            print(f"  Vector Search Top-100: Found={vector_found_100}, Rank={vector_rank_100}")

            results.append({
                'case_id': case_id,
                'syutugan': syutugan,
                'expected': expected,
                'vector_found_10': vector_found_10,
                'vector_rank_10': vector_rank_10,
                'vector_found_50': vector_found_50,
                'vector_rank_50': vector_rank_50,
                'vector_found_100': vector_found_100,
                'vector_rank_100': vector_rank_100,
                'vector_top5_10': vector_patent_ids_10[:5]
            })

            # レート制限対策
            time.sleep(0.5)

        # 結果をDataFrameに変換
        results_df = pd.DataFrame(results)

        # 統計を表示
        self.print_statistics(results_df)

        # 結果を保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(f'vector_evaluation_filtered_{timestamp}.csv', index=False)
        print(f"\nResults saved to: vector_evaluation_filtered_{timestamp}.csv")

        return results_df

    def print_statistics(self, results_df):
        """統計情報を表示"""
        total = len(results_df)

        if total == 0:
            print("No valid cases to evaluate")
            return

        vector_found_10 = results_df['vector_found_10'].sum()
        vector_found_50 = results_df['vector_found_50'].sum()
        vector_found_100 = results_df['vector_found_100'].sum()

        vector_top3_10 = sum(1 for r in results_df['vector_rank_10'] if 0 < r <= 3)
        vector_top5_10 = sum(1 for r in results_df['vector_rank_10'] if 0 < r <= 5)
        vector_top3_50 = sum(1 for r in results_df['vector_rank_50'] if 0 < r <= 3)
        vector_top5_50 = sum(1 for r in results_df['vector_rank_50'] if 0 < r <= 5)
        vector_top3_100 = sum(1 for r in results_df['vector_rank_100'] if 0 < r <= 3)
        vector_top5_100 = sum(1 for r in results_df['vector_rank_100'] if 0 < r <= 5)

        print("\n" + "="*80)
        print("FILTERED VECTOR SEARCH EVALUATION RESULTS")
        print("="*80)

        print(f"\nTotal filtered cases: {total}")

        print("\nVector Search Results:")
        print(f"  Top-10 Found: {vector_found_10}/{total} ({vector_found_10/total:.2%})")
        print(f"  Top-10 Top-3: {vector_top3_10}/{total} ({vector_top3_10/total:.2%})")
        print(f"  Top-10 Top-5: {vector_top5_10}/{total} ({vector_top5_10/total:.2%})")
        print(f"  Top-50 Found: {vector_found_50}/{total} ({vector_found_50/total:.2%})")
        print(f"  Top-50 Top-3: {vector_top3_50}/{total} ({vector_top3_50/total:.2%})")
        print(f"  Top-50 Top-5: {vector_top5_50}/{total} ({vector_top5_50/total:.2%})")
        print(f"  Top-100 Found: {vector_found_100}/{total} ({vector_found_100/total:.2%})")
        print(f"  Top-100 Top-3: {vector_top3_100}/{total} ({vector_top3_100/total:.2%})")
        print(f"  Top-100 Top-5: {vector_top5_100}/{total} ({vector_top5_100/total:.2%})")

        # 平均ランク（見つかった場合のみ）
        vector_ranks_10 = [r for r in results_df['vector_rank_10'] if r > 0]
        vector_ranks_50 = [r for r in results_df['vector_rank_50'] if r > 0]
        vector_ranks_100 = [r for r in results_df['vector_rank_100'] if r > 0]

        print("\nAverage Rank (when found):")
        if vector_ranks_10:
            print(f"  Vector Search Top-10: {np.mean(vector_ranks_10):.2f}")
        if vector_ranks_50:
            print(f"  Vector Search Top-50: {np.mean(vector_ranks_50):.2f}")
        if vector_ranks_100:
            print(f"  Vector Search Top-100: {np.mean(vector_ranks_100):.2f}")


def main():
    """メイン処理"""
    evaluator = FilteredVectorEvaluator()

    print("Starting filtered vector search evaluation")
    print("="*80)

    # フィルタされたケースでの評価実行
    evaluator.evaluate_filtered_cases()

    print("\nFiltered evaluation completed!")


if __name__ == "__main__":
    main()
