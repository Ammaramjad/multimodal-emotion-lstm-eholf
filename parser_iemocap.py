"""
IEMOCAP annotation parser supporting common formats (CSV, TXT, TextGrid).
Functions:
- load_annotations(path)
- load_iemocap_metadata(root_dir, emotion_map=None)
"""

import os
import glob
import csv
from typing import List, Dict, Optional

try:
    from textgrid import TextGrid
    _HAS_TEXTGRID = True
except Exception:
    _HAS_TEXTGRID = False

def parse_csv(path: str) -> List[Dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                start = float(r.get('start', r.get('t1', r.get('begin', 0))))
                end = float(r.get('end', r.get('t2', r.get('stop', 0))))
            except Exception:
                start = float(r.get('xmin', 0)) if r.get('xmin') else 0.0
                end = float(r.get('xmax', 0)) if r.get('xmax') else 0.0
            text = r.get('text', r.get('utterance', '')).strip()
            label = r.get('label', r.get('emotion', '')).strip()
            rows.append({'start': start, 'end': end, 'text': text, 'label': label})
    return rows

def parse_simple_txt(path: str) -> List[Dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 2)
            if len(parts) < 3:
                rows.append({'start': 0.0, 'end': 0.0, 'text': line, 'label': ''})
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
                text = parts[2].strip()
            except Exception:
                start, end = 0.0, 0.0
                text = line
            rows.append({'start': start, 'end': end, 'text': text, 'label': ''})
    return rows

def parse_textgrid(path: str) -> List[Dict]:
    if not _HAS_TEXTGRID:
        raise RuntimeError("TextGrid parsing requires `textgrid` package.")
    tg = TextGrid.fromFile(path)
    rows = []
    for tier in tg.tiers:
        if hasattr(tier, 'intervals') and len(tier.intervals) > 0:
            for iv in tier.intervals:
                text = iv.mark.strip() if iv.mark is not None else ""
                rows.append({'start': float(iv.minTime), 'end': float(iv.maxTime), 'text': text, 'label': ''})
            return rows
    return rows

def load_annotations(path: str) -> List[Dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.csv', '.tsv']:
        return parse_csv(path)
    if ext in ['.txt', '.utt']:
        return parse_simple_txt(path)
    if ext in ['.textgrid', '.TextGrid', '.tg']:
        return parse_textgrid(path)
    try:
        return parse_csv(path)
    except Exception:
        return parse_simple_txt(path)

def map_labels(ann_list: List[Dict], emotion_map: Optional[Dict[str,int]] = None) -> List[Dict]:
    if emotion_map is None:
        return ann_list
    out = []
    for a in ann_list:
        lab = a.get('label', '').lower().strip()
        mapped = emotion_map.get(lab, emotion_map.get(lab.title(), None))
        a2 = a.copy()
        a2['label_mapped'] = mapped
        out.append(a2)
    return out

def find_media_for_utterance(root_dir: str, session_pattern="Session*") -> List[Dict]:
    results = []
    session_dirs = sorted([d for d in glob.glob(os.path.join(root_dir, session_pattern)) if os.path.isdir(d)])
    for sess in session_dirs:
        wavs = glob.glob(os.path.join(sess, '**', '*.wav'), recursive=True)
        for w in wavs:
            base = os.path.splitext(w)[0]
            video = None
            for ext in ['.mp4', '.avi', '.mov', '.MOV']:
                cand = base + ext
                if os.path.exists(cand):
                    video = cand
                    break
            ann = None
            for ext in ['.csv', '.TextGrid', '.textgrid', '.txt', '.utt']:
                cand = base + ext
                if os.path.exists(cand):
                    ann = cand
                    break
            results.append({'wav': w, 'video': video, 'ann': ann, 'session': os.path.basename(sess)})
    return results

def load_iemocap_metadata(root_dir: str, emotion_map: Optional[Dict[str,int]] = None):
    items = find_media_for_utterance(root_dir)
    utterances = []
    for item in items:
        wav = item['wav']
        video = item['video']
        sess = item['session']
        ann = item['ann']
        if ann:
            entries = load_annotations(ann)
            entries = map_labels(entries, emotion_map)
            for e in entries:
                utterances.append({'wav': wav, 'video': video, 'start': e['start'], 'end': e['end'],
                                   'text': e.get('text', ''), 'label': e.get('label_mapped', e.get('label', '')),
                                   'session': sess})
        else:
            utterances.append({'wav': wav, 'video': video, 'start': 0.0, 'end': 0.0, 'text': '', 'label': None, 'session': sess})
    return utterances