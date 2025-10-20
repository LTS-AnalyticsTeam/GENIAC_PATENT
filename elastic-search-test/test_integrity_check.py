#!/usr/bin/env python3
"""
整合性チェックのテスト実行スクリプト
"""
import asyncio
import sys
from pathlib import Path

# パスを追加
sys.path.append(str(Path(__file__).parent))

from run_integrity_check import IntegrityChecker


async def test_integrity_check():
    """整合性チェックのテスト実行"""
    print("🔍 整合性チェックツールのテスト実行を開始します")

    try:
        # 整合性チェッカーを初期化
        checker = IntegrityChecker()

        # 小さなサンプルサイズでテスト実行
        results = await checker.run_comprehensive_check(sample_size=100)

        # 結果の表示
        print("\n" + "=" * 70)
        print("📊 整合性チェック結果")
        print("=" * 70)

        # 基本件数
        basic_counts = results.get("basic_counts", {})
        if "error" not in basic_counts:
            print(f"\n📈 基本統計:")
            print(f"   Cosmos DB総数: {basic_counts.get('cosmos_total', 0):,}件")
            print(f"   Elasticsearch総数: {basic_counts.get('elasticsearch_total', 0):,}件")
            print(f"   差分: {basic_counts.get('difference', 0):,}件")
            print(f"   一致率: {basic_counts.get('match_percentage', 0)}%")
            print(f"   接続状態: ES={basic_counts.get('connection_status', {}).get('elasticsearch_connected', 'Unknown')}")
        else:
            print(f"❌ 基本件数チェックエラー: {basic_counts['error']}")

        # サンプリング検証
        sample_validation = results.get("sample_validation", {})
        if "error" not in sample_validation:
            print(f"\n🔍 サンプリング検証:")
            print(f"   検証件数: {sample_validation.get('sample_size', 0):,}件")
            print(f"   有効: {sample_validation.get('valid_documents', 0):,}件")
            print(f"   無効: {sample_validation.get('invalid_documents', 0):,}件")
            print(f"   成功率: {sample_validation.get('success_rate', 0)}%")
        else:
            print(f"❌ サンプリング検証エラー: {sample_validation['error']}")

        # 失敗ドキュメント分析
        failed_analysis = results.get("failed_documents_analysis", {})
        if "error" not in failed_analysis:
            print(f"\n❌ 失敗ドキュメント分析:")
            print(f"   総失敗件数: {failed_analysis.get('total_failed', 0):,}件")
        else:
            print(f"❌ 失敗ドキュメント分析エラー: {failed_analysis['error']}")

        # 推奨事項
        recommendations = results.get("recommendations", [])
        if recommendations:
            print(f"\n💡 推奨事項:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")

        # 実行時間
        duration = results.get("duration_seconds", 0)
        print(f"\n⏱️ 実行時間: {duration:.1f}秒")

        print("\n" + "=" * 70)
        print("✅ 整合性チェックテストが完了しました！")
        print("=" * 70)

        return results

    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(test_integrity_check())
