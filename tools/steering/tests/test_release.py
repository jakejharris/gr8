import hashlib
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from steering import release


class ReleaseTests(unittest.TestCase):
    def fixture(self, root):
        payload = b'\0\0\x80?' * 6
        manifest = {'schema_version': 1, 'profile_name': 'test-v1',
                    'model': {'hidden_size': 3}, 'layers': [1, 2], 'alphas': [1.0, 1.0],
                    'directions': {'file': 'directions.f32', 'dtype': 'float32-le', 'byte_count': len(payload),
                                   'sha256': hashlib.sha256(payload).hexdigest()}}
        with open(os.path.join(root, 'manifest.json'), 'w') as f: json.dump(manifest, f)
        with open(os.path.join(root, 'directions.f32'), 'wb') as f: f.write(payload)
        return manifest, payload

    def test_deterministic_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, payload = self.fixture(d)
            a, b, out = (os.path.join(d, x) for x in ('a.safetensors', 'b.safetensors', 'out'))
            release.pack(d, a); release.pack(d, b)
            with open(a, 'rb') as fa, open(b, 'rb') as fb:
                self.assertEqual(fa.read(), fb.read())
            got = release.unpack(a, out)
            self.assertEqual(got, manifest)
            with open(os.path.join(out, 'directions.f32'), 'rb') as f:
                self.assertEqual(f.read(), payload)

    def test_rejects_tampering_and_extra_tensor(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(d); p = os.path.join(d, 'p.safetensors'); release.pack(d, p)
            with open(p, 'rb') as f: raw = bytearray(f.read())
            raw[-1] ^= 1
            bad = os.path.join(d, 'bad.safetensors')
            with open(bad, 'wb') as f: f.write(raw)
            with self.assertRaises(release.ProfileError): release.read(bad)


if __name__ == '__main__': unittest.main()
