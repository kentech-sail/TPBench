"""Prompt templates, alias checks, and matched deletion selection."""
import re
import hashlib
import random
from typing import Any
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from normalize import normalize_value, value_occurs_in_text, normalized_phrase_occurs
TURN_RE = re.compile(r"^\[T(\d+)\] (USER|SYSTEM): ", re.MULTILINE)

def is_alias_equivalent(old: str | None, new: str | None, dataset: str) -> bool:
    if not old or not new:
        return False
    no = normalize_value(old, dataset)
    nn = normalize_value(new, dataset)
    if not no or not nn:
        return False
    if no == nn:
        return True
    no_pre = no.split(',', 1)[0].strip()
    nn_pre = nn.split(',', 1)[0].strip()
    if no_pre and nn_pre and (no_pre == nn_pre):
        return True
    short = min(len(no_pre), len(nn_pre))
    if short >= 3 and (no_pre in nn_pre or nn_pre in no_pre):
        return True
    return False

def make_p1_user(context: str) -> str:
    return f"""Context:\n{context}\n\nQuestion: What was the user's initial goal at the beginning of this conversation? Reply with one short noun phrase (e.g. "book a hotel", "find a flight", "reserve a restaurant").\n\nRequired JSON schema:\n{{"value": <string>, "support": <verbatim span from context, <= 30 words>, "abstain": <true if you cannot tell, else false>}}\nReply with only the JSON object."""

def make_p3_user(context: str, slot_human: str) -> str:
    return f"""Context:\n{context}\n\nQuestion: After the most recent change visible in the context, what is the user's CURRENT value for {slot_human}? Reply with one short value.\n\nRequired JSON schema:\n{{"value": <string>, "support": <verbatim span from context, <= 30 words>, "abstain": <true if you cannot tell, else false>}}\nReply with only the JSON object."""

def normalize_task(goal: str) -> str:
    if not goal:
        return ''
    value = goal.strip()
    value = re.sub("^(can you |could you |please |i'?d like to |i want to |i need to |i'?m looking to )", '', value, flags=re.IGNORECASE)
    value = re.sub('[?.!]+\\s*$', '', value)
    return value.strip().lower()

def p3_user_prompt(context: str, slot_human: str) -> str:
    return f"""Context:\n{context}\n\nQuestion: In one short reply, state BOTH (a) the user's task in this conversation AND (b) the final value the user decided on for {slot_human}. Format your answer as: "<task>; {slot_human}: <final value>".\n\nRequired JSON schema:\n{{"value": <string in the format above>, "support": <verbatim span from context, <= 30 words>, "abstain": <true if you cannot tell, else false>}}\nReply with only the JSON object."""

def split_turns(context: str) -> tuple[str, list[dict[str, Any]]]:
    matches = list(TURN_RE.finditer(context))
    if not matches:
        raise ValueError('context has no [Tn] turns')
    header = context[:matches[0].start()]
    turns: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(context)
        turn_id = int(match.group(1))
        if turn_id != idx:
            raise ValueError(f'non-contiguous turn id {turn_id} at position {idx}')
        turns.append({'turn_id': turn_id, 'speaker': match.group(2), 'text': context[match.end():end].rstrip('\n'), 'block': context[match.start():end].rstrip('\n')})
    return (header, turns)

def render_subset(header: str, turns: list[dict[str, Any]], keep: list[int]) -> str:
    blocks = [turns[idx]['block'] for idx in sorted(keep)]
    return header + '\n'.join(blocks)

def prompt_parts(prompt_user: str) -> tuple[str, str]:
    marker = '\n\nQuestion:'
    position = prompt_user.find(marker)
    if position < 0:
        raise ValueError('prompt has no Question marker')
    return (prompt_user[:position], prompt_user[position:])

def evidence_turns(turns: list[dict[str, Any]], gold: str, dataset: str) -> list[int]:
    return [turn['turn_id'] for turn in turns if value_occurs_in_text(gold, turn['text'], dataset)]

def matched_control(turns: list[dict[str, Any]], targets: list[int], forbidden: set[int], old_value: str, new_value: str, dataset: str) -> list[int] | None:
    old_norm = normalize_value(old_value or '', dataset)
    new_norm = normalize_value(new_value or '', dataset)
    clean = []
    for turn in turns:
        turn_id = turn['turn_id']
        if turn_id in forbidden:
            continue
        normalized = normalize_value(turn['text'], dataset)
        if old_norm and normalized_phrase_occurs(old_norm, normalized):
            continue
        if new_norm and normalized_phrase_occurs(new_norm, normalized):
            continue
        clean.append(turn_id)
    picked: list[int] = []
    for target in sorted(targets):
        available = [turn_id for turn_id in clean if turn_id not in picked]
        same_speaker = [turn_id for turn_id in available if turns[turn_id]['speaker'] == turns[target]['speaker']]
        candidates = same_speaker
        if not candidates:
            return None
        target_words = len(turns[target]['text'].split())
        best = min(candidates, key=lambda turn_id: (abs(len(turns[turn_id]['text'].split()) - target_words), abs(turn_id - target), turn_id))
        picked.append(best)
    return sorted(picked)

def turn_speaker(i: int) -> str:
    return 'USER' if i % 2 == 0 else 'SYSTEM'

def render_turns(turns: list[str], indices: list[int]) -> str:
    indices = sorted(set(indices))
    parts = []
    for i in indices:
        if 0 <= i < len(turns):
            parts.append(f'[T{i}] {turn_speaker(i)}: {turns[i]}')
    return '\n'.join(parts)

def method_full_context(record: dict) -> str:
    return render_turns(record['turns'], list(range(len(record['turns']))))

def method_recency(record: dict, k: int) -> str:
    n = len(record['turns'])
    return render_turns(record['turns'], list(range(max(0, n - k), n)))

def method_random_seed42(record: dict, k: int) -> str:
    n = len(record['turns'])
    h = hashlib.sha256(f"42|{record['dialogue_id']}".encode('utf-8')).hexdigest()
    seed = int(h[:16], 16) % 2 ** 32
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(n), min(k, n)))
    return render_turns(record['turns'], indices)

def method_first_n(record: dict, k: int) -> str:
    n = len(record['turns'])
    return render_turns(record['turns'], list(range(0, min(k, n))))

def method_uniform_stride(record: dict, k: int) -> str:
    n = len(record['turns'])
    if n == 0 or k <= 0:
        return ''
    if k >= n:
        return render_turns(record['turns'], list(range(n)))
    step = n / k
    indices = sorted({min(n - 1, int(round(i * step))) for i in range(k)})
    return render_turns(record['turns'], indices)
