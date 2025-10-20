# 特許ベクトル検索評価方法の詳細

## 概要

このドキュメントでは、特許引用関係を用いたベクトル検索システムの評価方法について詳細に説明します。

## 評価データセット

### データソース

- **Cosmos DB**: 正解データ（Ground Truth）として使用
- **Elasticsearch**: 評価対象のインデックス（patent_vectors）
- **テストケース**: 62 件の特許引用関係（syutugan → ax_docs）

### データ構造

```
テストケース形式:
- case_id: テストケースID
- syutugan: 引用元特許（例: JP2015163142A）
- ax_docs: 引用先特許（例: JP2013123456A）
```

## 評価手法

### 1. Term Search（用語検索）

**目的**: reference.syutugan フィールドを使った直接的な引用関係検索

**クエリ例**:

```json
{
  "query": {
    "term": {
      "reference.syutugan": "JP2015163142A"
    }
  }
}
```

**評価指標**:

- 精度（Accuracy）: 正解を見つけられた割合
- ランク（Rank）: 正解が何位に現れるか

### 2. Vector Search（ベクトル検索）

**目的**: 特許要約の意味的類似性による関連特許検索

**手順**:

1. syutugan 特許の要約を Cosmos DB から取得
2. Azure OpenAI（text-embedding-3-large）で embedding 化
3. Elasticsearch でコサイン類似度検索

**クエリ例**:

```json
{
  "query": {
    "script_score": {
      "query": {"match_all": {}},
      "script": {
        "source": "cosineSimilarity(params.query_vector, 'summary_vector') + 1.0",
        "params": {
          "query_vector": [0.1, 0.2, ...]
        }
      }
    }
  }
}
```

**評価指標**:

- Top-K 精度: Top-10/50/100 での発見率
- 平均ランク: 発見された場合の平均順位

### 3. Hybrid Search（ハイブリッド検索）

#### 3.1 従来手法（Bool Should）

Term Search と Vector Search を bool should で組み合わせ

```json
{
  "query": {
    "bool": {
      "should": [
        {
          "term": {
            "reference.syutugan": {
              "value": "JP2015163142A",
              "boost": 10.0
            }
          }
        },
        {
          "script_score": {
            "query": {"match_all": {}},
            "script": {
              "source": "cosineSimilarity(params.query_vector, 'summary_vector') + 1.0",
              "params": {"query_vector": [...]}
            },
            "boost": 1.0
          }
        }
      ]
    }
  }
}
```

#### 3.2 RRF 手法（Reciprocal Rank Fusion）

Elasticsearch の RRF 機能を使用した高度なランク融合

```json
{
  "query": {
    "rrf": {
      "queries": [
        {
          "term": {
            "reference.syutugan": "JP2015163142A"
          }
        },
        {
          "script_score": {
            "query": {"match_all": {}},
            "script": {
              "source": "cosineSimilarity(params.query_vector, 'summary_vector') + 1.0",
              "params": {"query_vector": [...]}
            }
          }
        }
      ],
      "rank_constant": 60,
      "rank_window_size": 100
    }
  }
}
```

**RRF パラメータ**:

- `rank_constant`: ランク定数（デフォルト 60）
- `rank_window_size`: ランクウィンドウサイズ（デフォルト 100）

**RRF スコア計算式**:

```
RRF_score = Σ(1 / (rank_constant + rank_i))
```

## 評価指標

### 基本指標

1. **精度（Accuracy）**: 正解を発見できた割合

   ```
   Accuracy = 発見成功数 / 総テストケース数
   ```

2. **Top-K 精度**: 上位 K 件内での発見率

   ```
   Top-K Accuracy = Top-K内発見数 / 総テストケース数
   ```

3. **平均ランク（Mean Rank）**: 発見された場合の平均順位
   ```
   Mean Rank = Σ(発見時のランク) / 発見成功数
   ```

### 比較分析

1. **改善度**: Vector Search から Hybrid Search への改善率

   ```
   Improvement = (Hybrid_Found - Vector_Found) / Vector_Found × 100%
   ```

2. **ランク改善**: 平均ランクの改善
   ```
   Rank_Improvement = Vector_Mean_Rank - Hybrid_Mean_Rank
   ```

## データ品質の課題

### 発見された問題

1. **参照データの欠損**: reference.syutugan フィールドが多くのドキュメントで空
2. **データ同期問題**: Cosmos DB と Elasticsearch 間の不整合
3. **文書の欠損**: 62 件中 15 件で syutugan または ax_docs 文書が存在しない

### フィルタリング戦略

有効な評価のため、両方の文書が存在する 47 件に絞って評価を実施

## 評価結果の解釈

### 成功基準

- **Term Search**: 直接的な引用関係の検出精度
- **Vector Search**: 意味的類似性による関連性発見能力
- **Hybrid Search**: 両手法の統合による総合性能

### 期待される結果

1. Term Search は高精度だが、データ品質に依存
2. Vector Search は意味的関連性を捉えるが、精度は中程度
3. Hybrid Search（特に RRF）は両者の利点を活用し、最高性能を実現

## 実装上の注意点

### レート制限

- Azure OpenAI API の呼び出し間隔: 0.5 秒
- 大量評価時のタイムアウト対策

### エラーハンドリング

- RRF 非対応時のフォールバック機能
- ネットワークエラー時の再試行機能

### 結果保存

- JSON 形式: 詳細結果（全データ）
- CSV 形式: サマリー結果（統計情報）
- タイムスタンプ付きファイル名で履歴管理

## 今後の改善案

1. **データ品質向上**: reference.syutugan フィールドの完全性向上
2. **評価データ拡張**: より多くのテストケースの追加
3. **パラメータ最適化**: RRF パラメータのチューニング
4. **多様な類似度指標**: コサイン類似度以外の指標の検討
