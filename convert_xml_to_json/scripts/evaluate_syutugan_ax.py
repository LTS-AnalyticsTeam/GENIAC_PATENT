import os
import csv
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv():
        return None

import importlib.util


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module: {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate(csv_in: str, csv_out: str, limit: int = 300) -> Dict[str, Any]:
    load_dotenv()
    here = os.path.dirname(__file__)
    runner = _load_module(os.path.join(here, 'run_text_to_cosmos_csv.py'), 'runner')
    search_mod = _load_module(os.path.join(here, 'cosmos_patent_search.py'), 'cosmos')

    total = 0
    hit_cases = 0
    rows_out: List[Dict[str, Any]] = []

    with open(csv_in, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            case_id = row.get('case_id', '')
            syutugan = (row.get('syutugan') or '').strip()
            ax_docs = (row.get('ax_docs') or '').strip()

            # Fetch source doc and build text
            doc = runner.fetch_cosmos_doc_by_patent_id(syutugan)
            if not doc:
                rows_out.append({
                    'case_id': case_id,
                    'syutugan': syutugan,
                    'ax_docs': ax_docs,
                    'hit_count': 0,
                    'ax_hit': 0,
                    'note': 'not found in cosmos'
                })
                continue

            text = runner.build_text_from_doc(doc)
            # Always run ABC keywords
            kws = runner.extract_keywords_using_abc(text, topn=12)
            # Always use Azure Search-based predictor for FI/F-term
            fi_cands, ft_cands = runner.fi_fterm_candidates_using_llm_keywords(kws)

            # Search
            results = search_mod.search_cosmos(kws, fi_cands, ft_cands, limit_total=limit)
            pids = {str(r.get('patent_id', '')) for r in results}
            ax_hit = 1 if ax_docs and ax_docs in pids else 0
            if ax_hit:
                hit_cases += 1

            rows_out.append({
                'case_id': case_id,
                'syutugan': syutugan,
                'ax_docs': ax_docs,
                'hit_count': len(results),
                'ax_hit': ax_hit
            })

    # Write output
    with open(csv_out, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['case_id', 'syutugan', 'ax_docs', 'hit_count', 'ax_hit'])
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    return {'total': total, 'ax_hit': hit_cases}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate syutugan -> search -> ax_docs hit')
    parser.add_argument('--in_csv', default=os.path.join(os.path.dirname(__file__), '..', 'syutugan_ax_test_data_converted.csv'))
    parser.add_argument('--out_csv', default='eval_syutugan_ax.csv', help='Single output (when not using batching)')
    parser.add_argument('--out_prefix', default='eval_syutugan_ax_part', help='Batch output prefix (used when --batch_size>0)')
    parser.add_argument('--batch_size', type=int, default=0, help='If >0, write CSV every N rows to <prefix>_NNN.csv')
    parser.add_argument('--limit', type=int, default=300)
    args = parser.parse_args()

    if args.batch_size and args.batch_size > 0:
        # Batch processing: write a CSV every batch_size rows
        load_dotenv()
        here = os.path.dirname(__file__)
        runner = _load_module(os.path.join(here, 'run_text_to_cosmos_csv.py'), 'runner')
        search_mod = _load_module(os.path.join(here, 'cosmos_patent_search.py'), 'cosmos')

        total = 0
        hit_cases = 0
        part_idx = 0
        buffer: List[Dict[str, Any]] = []

        def flush():
            nonlocal part_idx, buffer
            if not buffer:
                return
            part_idx += 1
            out_path = f"{args.out_prefix}_{part_idx:03d}.csv"
            with open(out_path, 'w', newline='', encoding='utf-8-sig') as wf:
                writer = csv.DictWriter(wf, fieldnames=['case_id', 'syutugan', 'ax_docs', 'hit_count', 'ax_hit'])
                writer.writeheader()
                for r in buffer:
                    writer.writerow(r)
            print(f"Wrote {len(buffer)} rows -> {out_path}")
            buffer = []

        with open(args.in_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                case_id = row.get('case_id', '')
                syutugan = (row.get('syutugan') or '').strip()
                ax_docs = (row.get('ax_docs') or '').strip()

                total += 1
                # Fetch and build text
                doc = runner.fetch_cosmos_doc_by_patent_id(syutugan)
                if not doc:
                    buffer.append({'case_id': case_id, 'syutugan': syutugan, 'ax_docs': ax_docs, 'hit_count': 0, 'ax_hit': 0})
                else:
                    text = runner.build_text_from_doc(doc)
                    kws = runner.extract_keywords_using_abc(text, topn=12)
                    fi_cands, ft_cands = runner.fi_fterm_candidates_using_llm_keywords(kws)
                    results = search_mod.search_cosmos(kws, fi_cands, ft_cands, limit_total=args.limit)
                    pids = {str(r.get('patent_id', '')) for r in results}
                    ax_hit = 1 if ax_docs and ax_docs in pids else 0
                    if ax_hit:
                        hit_cases += 1
                    buffer.append({'case_id': case_id, 'syutugan': syutugan, 'ax_docs': ax_docs, 'hit_count': len(results), 'ax_hit': ax_hit})

                if len(buffer) >= args.batch_size:
                    flush()

        # Flush remainder
        flush()
        print(f"Total cases: {total}")
        print(f"ax_docs hit (1): {hit_cases}")
    else:
        stats = evaluate(args.in_csv, args.out_csv, limit=args.limit)
        print(f"Total cases: {stats['total']}")
        print(f"ax_docs hit (1): {stats['ax_hit']}")


if __name__ == '__main__':
    main()
