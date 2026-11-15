import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spanver.evaluate import read_jsonl
from spanver.module import predict_type

LOCALIZED = "rufacts_localized.jsonl"
PAIRLEVEL = "rufacts_pairlevel.jsonl"
NEGATIVE = "rufacts_negative.jsonl"


CLAMPED = []


def doc_id(evidence):
    return hashlib.sha256((evidence or "").encode("utf-8")).hexdigest()[:16]


def facts_of(record):
    facts = record.get("claim_facts") or {}
    if isinstance(facts, dict):
        return [facts[key] for key in sorted(facts)]
    if isinstance(facts, list):
        return facts
    return []


def convert_spans(claim, record):
    spans = []
    used = []
    for fact in facts_of(record):
        if not isinstance(fact, dict):
            return None, "bad_fact"
        rng = fact.get("range") or []
        text = fact.get("span")
        if len(rng) != 2 or not isinstance(text, str):
            return None, "bad_range"
        start, end = int(rng[0]), int(rng[1])
        if 0 <= start and end <= len(claim) and end > start:
            pass
        elif 0 <= start and claim[start : start + len(text)] == text:
            end = start + len(text)
            CLAMPED.append(record.get("idx"))
        else:
            return None, "range_out_of_bounds"
        if claim[start:end] != text:
            return None, "span_mismatch"
        if any(not (end - 1 < left or start > right) for left, right in used):
            continue
        used.append((start, end - 1))
        spans.append({"start": start, "end": end - 1, "text": text, "type": predict_type(text, 0.0)})
    return spans, ""


def convert_record(record):
    evidence = record.get("evidence") or ""
    claim = record.get("claim") or ""
    if not evidence.strip() or not claim.strip():
        return None, "empty_text"
    label = int(record.get("label", 1))
    spans, reason = convert_spans(claim, record)
    if reason:
        return None, reason
    pair = {
        "pair_id": f"rufacts{int(record.get('idx', 0)):06d}",
        "source": "rufacts",
        "domain": "rufacts",
        "doc_id": doc_id(evidence),
        "document_text": evidence,
        "model_output": claim,
        "spans": spans,
        "label": label,
        "split": "train",
    }
    if spans:
        return "localized", pair
    if label == 1:
        return "pairlevel", pair
    return "negative", pair


def convert_files(paths, out_dir, limit=0, report_path=""):
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    buckets = {"localized": [], "pairlevel": [], "negative": []}
    reasons = {}
    total = 0
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    record = json.loads(line)
                except ValueError:
                    reasons["bad_json"] = reasons.get("bad_json", 0) + 1
                    continue
                name, payload = convert_record(record)
                if name is None:
                    reasons[payload] = reasons.get(payload, 0) + 1
                    continue
                buckets[name].append(payload)
    if limit:
        for name in buckets:
            buckets[name] = buckets[name][:limit]
    for name, filename in (("localized", LOCALIZED), ("pairlevel", PAIRLEVEL), ("negative", NEGATIVE)):
        with open(target / filename, "w", encoding="utf-8") as handle:
            for pair in buckets[name]:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    report = {
        "inputs": [str(path) for path in paths],
        "records": total,
        "localized": len(buckets["localized"]),
        "pairlevel": len(buckets["pairlevel"]),
        "negative": len(buckets["negative"]),
        "rejected": sum(reasons.values()),
        "reasons": reasons,
        "clamped": len(CLAMPED),
        "clamped_idx": sorted(CLAMPED),
        "spans": sum(len(pair["spans"]) for pair in buckets["localized"]),
        "files": [str(target / LOCALIZED), str(target / PAIRLEVEL), str(target / NEGATIVE)],
    }
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_extra(localized, our_pairs, count=200):
    taken = {pair.get("doc_id") or doc_id(pair.get("document_text") or "") for pair in our_pairs}
    chosen = []
    overlap = 0
    for pair in localized:
        if len(chosen) >= count:
            break
        if pair["doc_id"] in taken:
            overlap += 1
            continue
        taken.add(pair["doc_id"])
        item = dict(pair)
        item["split"] = "extra"
        chosen.append(item)
    return chosen, overlap


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inputs", nargs="+", required=True)
    parser.add_argument("--out", default="artifacts/rufacts")
    parser.add_argument("--report", default="artifacts/rufacts_report.json")
    parser.add_argument("--extra", type=int, default=0)
    parser.add_argument("--our", default="artifacts/gold.jsonl")
    parser.add_argument("--gold-out", default="artifacts/gold_1200.jsonl")
    args = parser.parse_args(argv)

    report = convert_files(args.inputs, args.out, report_path=args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    if args.extra:
        localized = read_jsonl(Path(args.out) / LOCALIZED)
        ours = read_jsonl(Path(args.our))
        chosen, overlap = build_extra(localized, ours, args.extra)
        merged = list(ours) + chosen
        Path(args.gold_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.gold_out, "w", encoding="utf-8") as handle:
            for pair in merged:
                handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
        extra_report = {
            "gold_out": args.gold_out,
            "our_pairs": len(ours),
            "extra_pairs": len(chosen),
            "total": len(merged),
            "overlap_excluded": overlap,
        }
        print(json.dumps(extra_report, ensure_ascii=False, indent=2), flush=True)
        target = Path(args.report)
        base = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
        base["extra"] = extra_report
        target.write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
