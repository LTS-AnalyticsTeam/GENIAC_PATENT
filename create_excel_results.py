"""
評価結果をExcel形式で出力するスクリプト
当たっていたら1、当たっていなかったら0で表示
"""

from datetime import datetime

import pandas as pd


def create_excel_results():
    """CSVファイルを読み込んでExcel形式で結果を出力"""

    # CSVファイルを読み込み
    df = pd.read_csv('evaluation_summary_extended_20250817_224209.csv')

    # 新しいDataFrameを作成（1/0形式）
    results_df = pd.DataFrame()

    # 基本情報
    results_df['case_id'] = df['case_id']
    results_df['syutugan'] = df['syutugan']
    results_df['expected'] = df['expected']

    # Cosmos DB結果（1/0）
    results_df['cosmos_found'] = df['cosmos_found'].astype(int)

    # Elasticsearch結果（1/0）
    results_df['es_found'] = df['es_found'].astype(int)

    # Vector Search結果（1/0）
    results_df['vector_found_10'] = df['vector_found_10'].astype(int)
    results_df['vector_found_50'] = df['vector_found_50'].astype(int)
    results_df['vector_found_100'] = df['vector_found_100'].astype(int)

    # Hybrid Search結果（1/0）
    results_df['hybrid_found_10'] = df['hybrid_found_10'].astype(int)
    results_df['hybrid_found_50'] = df['hybrid_found_50'].astype(int)
    results_df['hybrid_found_100'] = df['hybrid_found_100'].astype(int)

    # ランク情報も含める（参考用）
    results_df['es_rank'] = df['es_rank']
    results_df['vector_rank_10'] = df['vector_rank_10']
    results_df['vector_rank_50'] = df['vector_rank_50']
    results_df['vector_rank_100'] = df['vector_rank_100']
    results_df['hybrid_rank_10'] = df['hybrid_rank_10']
    results_df['hybrid_rank_50'] = df['hybrid_rank_50']
    results_df['hybrid_rank_100'] = df['hybrid_rank_100']

    # タイムスタンプ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Excelファイルに出力
    excel_filename = f'patent_evaluation_results_{timestamp}.xlsx'

    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # メイン結果シート
        results_df.to_excel(writer, sheet_name='Results', index=False)

        # 統計サマリーシート
        summary_df = create_summary_sheet(results_df)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # 詳細分析シート
        analysis_df = create_analysis_sheet(results_df)
        analysis_df.to_excel(writer, sheet_name='Analysis', index=False)

    print(f"Excel file created: {excel_filename}")
    return excel_filename

def create_summary_sheet(results_df):
    """統計サマリーシートを作成"""
    total_cases = len(results_df)

    summary_data = []

    # 各手法の統計
    methods = [
        ('Cosmos DB', 'cosmos_found'),
        ('Elasticsearch', 'es_found'),
        ('Vector Search Top-10', 'vector_found_10'),
        ('Vector Search Top-50', 'vector_found_50'),
        ('Vector Search Top-100', 'vector_found_100'),
        ('Hybrid Search Top-10', 'hybrid_found_10'),
        ('Hybrid Search Top-50', 'hybrid_found_50'),
        ('Hybrid Search Top-100', 'hybrid_found_100')
    ]

    for method_name, column in methods:
        found_count = results_df[column].sum()
        accuracy = found_count / total_cases

        summary_data.append({
            'Method': method_name,
            'Found': found_count,
            'Total': total_cases,
            'Accuracy': f"{accuracy:.2%}",
            'Accuracy_Decimal': accuracy
        })

    return pd.DataFrame(summary_data)

def create_analysis_sheet(results_df):
    """詳細分析シートを作成"""
    analysis_data = []

    # 各ケースの分析
    for _, row in results_df.iterrows():
        case_analysis = {
            'case_id': row['case_id'],
            'syutugan': row['syutugan'],
            'expected': row['expected'],
            'cosmos_hit': row['cosmos_found'],
            'es_hit': row['es_found'],
            'vector_10_hit': row['vector_found_10'],
            'vector_50_hit': row['vector_found_50'],
            'vector_100_hit': row['vector_found_100'],
            'hybrid_10_hit': row['hybrid_found_10'],
            'hybrid_50_hit': row['hybrid_found_50'],
            'hybrid_100_hit': row['hybrid_found_100'],
            'total_hits': (row['cosmos_found'] + row['es_found'] +
                          row['vector_found_10'] + row['vector_found_50'] + row['vector_found_100'] +
                          row['hybrid_found_10'] + row['hybrid_found_50'] + row['hybrid_found_100']),
            'best_method': get_best_method(row),
            'vector_improvement': row['vector_found_100'] - row['vector_found_10'],
            'hybrid_vs_vector': row['hybrid_found_10'] - row['vector_found_10']
        }
        analysis_data.append(case_analysis)

    return pd.DataFrame(analysis_data)

def get_best_method(row):
    """最も良い結果を出した手法を特定"""
    methods = {
        'Cosmos DB': row['cosmos_found'],
        'Elasticsearch': row['es_found'],
        'Vector Top-10': row['vector_found_10'],
        'Vector Top-50': row['vector_found_50'],
        'Vector Top-100': row['vector_found_100'],
        'Hybrid Top-10': row['hybrid_found_10'],
        'Hybrid Top-50': row['hybrid_found_50'],
        'Hybrid Top-100': row['hybrid_found_100']
    }

    best_methods = [method for method, result in methods.items() if result == 1]

    if not best_methods:
        return "None"
    elif len(best_methods) == 1:
        return best_methods[0]
    else:
        return ", ".join(best_methods)

if __name__ == "__main__":
    create_excel_results()
