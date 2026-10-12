#!/usr/bin/env python3
"""Pack and unpack a GR8 schema-v1 steering profile as one deterministic safetensors file.

The payload has one F32 tensor, ``directions``.  The exact profile manifest is stored as canonical JSON in
``__metadata__.gr8_manifest``.  This tool intentionally uses only the Python standard library so the launcher can
materialize a downloaded profile before any model dependencies are installed.
"""
import argparse
import hashlib
import json
import os
import re
import struct

FORMAT = 'gr8-steering-profile-v1'
MAX_HEADER = 1 << 20
MAX_PAYLOAD = 64 << 20


class ProfileError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def validate_manifest(manifest, payload):
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1:
        raise ProfileError('want a schema-v1 steering manifest')
    name = manifest.get('profile_name')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', name):
        raise ProfileError('invalid profile_name')
    layers, alphas = manifest.get('layers'), manifest.get('alphas')
    model, directions = manifest.get('model'), manifest.get('directions')
    if not isinstance(layers, list) or not layers or not isinstance(alphas, list) or len(layers) != len(alphas):
        raise ProfileError('inconsistent layers and alphas')
    if not isinstance(model, dict) or not isinstance(model.get('hidden_size'), int) or model['hidden_size'] <= 0:
        raise ProfileError('invalid model.hidden_size')
    want = len(layers) * model['hidden_size'] * 4
    if not isinstance(directions, dict) or directions.get('file') != 'directions.f32' or \
            directions.get('dtype') != 'float32-le' or directions.get('byte_count') != want:
        raise ProfileError('invalid directions record')
    if len(payload) != want or len(payload) > MAX_PAYLOAD:
        raise ProfileError('directions payload size mismatch')
    if directions.get('sha256') != sha256(payload):
        raise ProfileError('directions payload digest mismatch')
    return name, [len(layers), model['hidden_size']]


def pack(profile_dir, output):
    with open(os.path.join(profile_dir, 'manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    with open(os.path.join(profile_dir, 'directions.f32'), 'rb') as f:
        payload = f.read(MAX_PAYLOAD + 1)
    _, shape = validate_manifest(manifest, payload)
    manifest_json = canonical(manifest)
    header = {
        '__metadata__': {'format': FORMAT, 'gr8_manifest': manifest_json,
                         'manifest_sha256': sha256(manifest_json.encode())},
        'directions': {'dtype': 'F32', 'shape': shape, 'data_offsets': [0, len(payload)]},
    }
    raw_header = canonical(header).encode()
    raw_header += b' ' * ((8 - len(raw_header) % 8) % 8)
    if len(raw_header) > MAX_HEADER:
        raise ProfileError('safetensors header is too large')
    tmp = output + '.tmp'
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(tmp, 'wb') as f:
        f.write(struct.pack('<Q', len(raw_header)))
        f.write(raw_header)
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, output)


def read(path):
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ProfileError('truncated safetensors header length')
        n, = struct.unpack('<Q', raw)
        if n <= 0 or n > MAX_HEADER or 8 + n > size:
            raise ProfileError('invalid safetensors header length')
        try:
            header = json.loads(f.read(n))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ProfileError(f'invalid safetensors header: {e}') from None
        if set(header) != {'__metadata__', 'directions'}:
            raise ProfileError('want exactly one directions tensor')
        meta, tensor = header['__metadata__'], header['directions']
        if not isinstance(meta, dict) or meta.get('format') != FORMAT or not isinstance(meta.get('gr8_manifest'), str):
            raise ProfileError('missing GR8 steering metadata')
        if not isinstance(tensor, dict) or set(tensor) != {'dtype', 'shape', 'data_offsets'} or tensor['dtype'] != 'F32':
            raise ProfileError('directions must be one F32 tensor')
        offsets = tensor['data_offsets']
        if not isinstance(offsets, list) or offsets != [0, size - 8 - n] or offsets[1] > MAX_PAYLOAD:
            raise ProfileError('invalid directions data offsets')
        payload = f.read()
    try:
        manifest = json.loads(meta['gr8_manifest'])
    except json.JSONDecodeError as e:
        raise ProfileError(f'invalid embedded manifest: {e}') from None
    manifest_json = canonical(manifest)
    if meta.get('gr8_manifest') != manifest_json or meta.get('manifest_sha256') != sha256(manifest_json.encode()):
        raise ProfileError('embedded manifest is not canonical or has the wrong digest')
    _, shape = validate_manifest(manifest, payload)
    if tensor['shape'] != shape:
        raise ProfileError('directions tensor shape mismatch')
    return manifest, payload


def unpack(path, output_dir):
    manifest, payload = read(path)
    os.makedirs(output_dir, exist_ok=True)
    files = {'manifest.json': (canonical(manifest) + '\n').encode(), 'directions.f32': payload}
    for name, data in files.items():
        dest = os.path.join(output_dir, name)
        with open(dest + '.tmp', 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(dest + '.tmp', dest)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('pack'); p.add_argument('profile_dir'); p.add_argument('output')
    p = sub.add_parser('unpack'); p.add_argument('input'); p.add_argument('output_dir')
    args = ap.parse_args(argv)
    if args.command == 'pack':
        pack(args.profile_dir, args.output)
    else:
        unpack(args.input, args.output_dir)


if __name__ == '__main__':
    main()
