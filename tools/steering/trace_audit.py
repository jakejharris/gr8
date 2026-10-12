#!/usr/bin/env python3
"""Compare normalized CUDA traces and enforce the steering zero-hot-path-overhead contract."""
import argparse
import collections
import json


def load(path):
    out = collections.Counter()
    meta = {}
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            arm = r.pop('arm'); count = int(r.pop('count', 1))
            key = (arm, tuple(sorted(r.items())))
            out[key] += count
            meta.setdefault(arm, set()).add(tuple(sorted(r.items())))
    return out, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--baseline', default='Z'); p.add_argument('--arms', default='P,R,M')
    a = p.parse_args(); counts, meta = load(a.input); baseline = a.baseline; failures = []
    for arm in a.arms.split(','):
        keys = meta.get(baseline, set()) | meta.get(arm, set())
        for key in sorted(keys):
            z, v = counts[(baseline, key)], counts[(arm, key)]
            fields = dict(key)
            if fields.get('kind') in ('kernel', 'memcpy', 'allocation', 'synchronization', 'graph') and z != v:
                failures.append({'arm': arm, 'baseline': z, 'candidate': v, **fields})
    result = {'schema': 'g6-steering-trace-audit-v1', 'baseline': baseline, 'arms': a.arms.split(','),
              'pass': not failures, 'differences': failures}
    with open(a.output, 'w') as f: json.dump(result, f, indent=2, sort_keys=True); f.write('\n')
    print('PASS' if not failures else f'FAIL: {len(failures)} trace differences')
    raise SystemExit(bool(failures))


if __name__ == '__main__': main()
