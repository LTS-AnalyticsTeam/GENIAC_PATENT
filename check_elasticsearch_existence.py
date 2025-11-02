"""
評価用CSVのsyutuganとax_docsがElasticsearchに存在するかを確認するスクリプト
"""

import os

import pandas as pd
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# 環境変数の読み込み
load_dotenv()

def check_elasticsearch_existence():
    """Elasticsearchでsyutuganとax_docsの存在を確認"""

    # Elasticsearch接続
    es = Elasticsearch('http://localhost:9200')
    index_name = 'patent_vectors'

    # テストケースを読み込み
    df = pd.read_csv('filtered_test_cases_20250813_174350.csv')
    print(f"Loaded {len(df)} test cases")

    results = []

    for idx, row in df.iterrows():
        case_id = row['case_id']
        syutugan = row['syutugan']
        ax_docs = row['ax_docs']

        print(f"\nChecking {case_id}: syutugan={syutugan}, ax_docs={ax_docs}")

        # 1. syutuganがElasticsearchに存在するかチェック
        syutugan_patent_num = syutugan.replace('JP', '').replace('A', '')
        syutugan_exists = check_document_exists(es, index_name, syutugan_patent_num)

        # 2. ax_docsがElasticsearchに存在するかチェック
        ax_docs_patent_num = ax_docs.replace('JP', '').replace('A', '')
        ax_docs_exists = check_document_exists(es, index_name, ax_docs_patent_num)

        # 3. syutuganのreference.syutuganフィールドにax_docsが含まれているかチェック
        reference_contains = check_reference_contains(es, index_name, syutugan, ax_docs)

        print(f"  syutugan exists: {syutugan_exists}")
        print(f"  ax_docs exists: {ax_docs_exists}")
        print(f"  reference contains: {reference_contains}")

        results.append({
            'case_id': case_id,
            'syutugan': syutugan,
            'ax_docs': ax_docs,
            'syutugan_exists': syutugan_exists,
            'ax_docs_exists': ax_docs_exists,
            'reference_contains': reference_contains,
            'both_exist': syutugan_exists and ax_docs_exists
        })

    # 結果をDataFrameに変換
    results_df = pd.DataFrame(results)

    # 統計を表示
    print_statistics(results_df)

    # 結果をCSVに保存
    results_df.to_csv('elasticsearch_existence_check.csv', index=False)
    print("\nResults saved to: elasticsearch_existence_check.csv")

    return results_df

def check_document_exists(es, index_name, patent_num):
    """指定された特許番号のドキュメントがElasticsearchに存在するかチェック"""
    try:
        response = es.get(index=index_name, id=patent_num)
        return True
    except Exception:
        return False

def check_reference_contains(es, index_name, syutugan, ax_docs):
    """syutuganのreference.syutuganフィールドにax_docsが含まれているかチェック"""
    try:
        syutugan_patent_num = syutugan.replace('JP', '').replace('A', '')
        response = es.get(index=index_name, id=syutugan_patent_num)

        if response['found']:
            reference_syutugan = response['_source'].get('reference', {}).get('syutugan', [])
            return ax_docs in reference_syutugan
        else:
            return False
    except Exception as e:
        print(f"    Error checking reference: {e}")
        return False

def print_statistics(results_df):
    """統計情報を表示"""
    total = len(results_df)

    syutugan_exists_count = results_df['syutugan_exists'].sum()
    ax_docs_exists_count = results_df['ax_docs_exists'].sum()
    reference_contains_count = results_df['reference_contains'].sum()
    both_exist_count = results_df['both_exist'].sum()

    print("\n" + "="*60)
    print("ELASTICSEARCH EXISTENCE CHECK RESULTS")
    print("="*60)

    print(f"\nTotal test cases: {total}")
    print(f"syutugan exists in Elasticsearch: {syutugan_exists_count}/{total} ({syutugan_exists_count/total:.2%})")
    print(f"ax_docs exists in Elasticsearch: {ax_docs_exists_count}/{total} ({ax_docs_exists_count/total:.2%})")
    print(f"reference.syutugan contains ax_docs: {reference_contains_count}/{total} ({reference_contains_count/total:.2%})")
    print(f"Both syutugan and ax_docs exist: {both_exist_count}/{total} ({both_exist_count/total:.2%})")

    # 存在しないケースを表示
    missing_syutugan = results_df[~results_df['syutugan_exists']]
    missing_ax_docs = results_df[~results_df['ax_docs_exists']]
    missing_reference = results_df[~results_df['reference_contains']]

    if len(missing_syutugan) > 0:
        print(f"\nMissing syutugan documents ({len(missing_syutugan)} cases):")
        for _, row in missing_syutugan.iterrows():
            print(f"  {row['case_id']}: {row['syutugan']}")

    if len(missing_ax_docs) > 0:
        print(f"\nMissing ax_docs documents ({len(missing_ax_docs)} cases):")
        for _, row in missing_ax_docs.iterrows():
            print(f"  {row['case_id']}: {row['ax_docs']}")

    if len(missing_reference) > 0:
        print(f"\nMissing reference relationships ({len(missing_reference)} cases):")
        for _, row in missing_reference.iterrows():
            print(f"  {row['case_id']}: {row['syutugan']} -> {row['ax_docs']}")

if __name__ == "__main__":
    check_elasticsearch_existence()
