import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.audit50 import verified_rows


def main():
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=range(5), required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rows = verified_rows()
    base = ROOT / 'models/rubert-tiny2'
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(str(base), local_files_only=True, attn_implementation='eager')
    # One epoch over every full reference and generated_text, in 254-token blocks.
    # Only these two text fields are read: gold and silver annotations are never used.
    blocks = []
    field_tokens = {}
    for field in ('references', 'generated_text'):
        total = 0
        for row in rows:
            ids = tokenizer(row[field], add_special_tokens=False, truncation=False)['input_ids']
            total += len(ids)
            for start in range(0, len(ids), 254):
                piece = ids[start:start + 254]
                if piece:
                    blocks.append({'input_ids': tokenizer.build_inputs_with_special_tokens(piece)})
        field_tokens[field] = total
    if not blocks:
        raise ValueError('нет текстов для MLM')
    random.Random(args.seed).shuffle(blocks)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    model.train()
    steps = 0
    loss_total = 0.0
    masked_total = 0
    for offset in range(0, len(blocks), 4):
        batch = collator(blocks[offset:offset + 4])
        mask = batch['labels'] != -100
        if not mask.any():
            continue
        optim.zero_grad(set_to_none=True)
        hidden = model.bert(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], return_dict=True).last_hidden_state
        logits = model.cls(hidden[mask])
        loss = torch.nn.functional.cross_entropy(logits, batch['labels'][mask])
        if not torch.isfinite(loss):
            raise ValueError('нечисловой MLM loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        count = int(mask.sum())
        loss_total += float(loss.detach()) * count
        masked_total += count
        steps += 1
        if steps % 50 == 0:
            print(f'seed {args.seed}: {offset + len(batch["input_ids"])}/{len(blocks)} блоков, MLM loss {loss_total/masked_total}', flush=True)
    folder = ROOT / f'weights/tapt50/seed_{args.seed}'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'encoder.pt'
    torch.save(model.bert.state_dict(), output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {'seed': args.seed, 'sha256': digest, 'bytes': output.stat().st_size, 'source_sha256': '21464872b18b803959525063753fb8e92937b1dee0099a6e1bbdd7832c411b21', 'rows': len(rows), 'fields': ['references', 'generated_text'], 'gold_used': False, 'silver_used': False, 'epochs': 1, 'lr': 1e-5, 'weight_decay': 0.01, 'mask_probability': 0.15, 'max_length': 256, 'batch_size': 4, 'blocks': len(blocks), 'tokens_by_field': field_tokens, 'steps': steps, 'masked_tokens': masked_total, 'mean_mlm_loss': loss_total / masked_total, 'elapsed_seconds': time.perf_counter() - started}
    (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
