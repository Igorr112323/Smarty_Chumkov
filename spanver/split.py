import hashlib

TEST = "test"
VAL = "val"
TRAIN = "train"
UNUSED = "unused"
SPLITS = (TEST, VAL, TRAIN)
PRIORITY = (TEST, VAL, TRAIN)
FRACTIONS = {TEST: 0.30, VAL: 0.15, TRAIN: 0.55}


def components(pairs):
    adjacency = {pair["pair_id"]: set() for pair in pairs}
    documents = {}
    for pair in pairs:
        for doc in pair.get("doc_ids") or []:
            documents.setdefault(doc, []).append(pair["pair_id"])
    for ids in documents.values():
        ordered_ids = sorted(set(ids))
        if len(ordered_ids) < 2:
            continue
        anchor = ordered_ids[0]
        for other in ordered_ids[1:]:
            adjacency[anchor].add(other)
            adjacency[other].add(anchor)
    seen = set()
    result = []
    for pair_id in sorted(adjacency):
        if pair_id in seen:
            continue
        seen.add(pair_id)
        stack = [pair_id]
        members = []
        while stack:
            current = stack.pop()
            members.append(current)
            for neighbour in sorted(adjacency[current], reverse=True):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        result.append(sorted(members))
    return result


def component_key(members):
    digest = hashlib.sha1(min(members).encode("utf-8")).hexdigest()
    return (len(members), digest)


def targets(pairs):
    manual = sum(1 for pair in pairs if pair.get("label_source") == "manual")
    test = int(round(manual * FRACTIONS[TEST]))
    val = int(round(manual * FRACTIONS[VAL]))
    train = manual - test - val
    return {TEST: test, VAL: val, TRAIN: train}


def assign(pairs):
    by_id = {pair["pair_id"]: pair for pair in pairs}
    manual_ids = {pair["pair_id"] for pair in pairs if pair.get("label_source") == "manual"}
    goal = targets(pairs)
    filled = {TEST: 0, VAL: 0, TRAIN: 0}
    assignment = {}
    groups = []
    for members in components(pairs):
        weight = sum(1 for member in members if member in manual_ids)
        groups.append((component_key(members), members, weight))
    groups.sort(key=lambda item: item[0])
    for _, members, weight in groups:
        options = [TEST, VAL, TRAIN] if weight > 0 else [TRAIN, VAL]
        best = None
        best_gap = None
        for split in options:
            gap = goal[split] - filled[split] - weight
            if best is None or gap > best_gap or (gap == best_gap and PRIORITY.index(split) < PRIORITY.index(best)):
                best = split
                best_gap = gap
        assignment_key = best
        for member in members:
            assignment[member] = assignment_key
        filled[assignment_key] += weight
    result = {}
    empty_ids = 0
    for pair in pairs:
        pair_id = pair["pair_id"]
        if not pair.get("doc_ids"):
            result[pair_id] = UNUSED
            empty_ids += 1
            continue
        chosen = assignment.get(pair_id, UNUSED)
        if chosen == TEST and pair_id not in manual_ids:
            result[pair_id] = UNUSED
        else:
            result[pair_id] = chosen
    counts = {TEST: 0, VAL: 0, TRAIN: 0, UNUSED: 0}
    manual_counts = {TEST: 0, VAL: 0, TRAIN: 0, UNUSED: 0}
    test_with_spans = 0
    for pair in pairs:
        split = result[pair["pair_id"]]
        counts[split] += 1
        if pair.get("label_source") == "manual":
            manual_counts[split] += 1
        if split == TEST and pair.get("spans"):
            test_with_spans += 1
    sizes = [len(members) for members in components(pairs)]
    stats = {
        "components": len(sizes),
        "max_component": max(sizes) if sizes else 0,
        "split_counts": counts,
        "split_manual": manual_counts,
        "test_pairs_with_spans": test_with_spans,
        "empty_doc_ids": empty_ids,
        "unused_unlabeled": sum(
            1
            for pair in pairs
            if result[pair["pair_id"]] == UNUSED and pair.get("label_source") != "manual"
        ),
        "targets": goal,
    }
    return result, stats


def doc_overlap(pairs, assignment):
    owners = {}
    for pair in pairs:
        split = assignment.get(pair["pair_id"])
        if split not in SPLITS:
            continue
        for doc in pair.get("doc_ids") or []:
            owners.setdefault(doc, set()).add(split)
    return sum(1 for splits in owners.values() if len(splits) > 1)
