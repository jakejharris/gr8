import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from steering import release


class ReleaseTests(unittest.TestCase):
    def fixture(self, root):
        payload = struct.pack('<6f', 1, 0, 0, 0, 1, 0)
        manifest = {'schema_version': 1, 'profile_name': 'test-v1',
                    'model': {'name': 'glm-5.3', 'revision': 'rev', 'weight_set': 'a' * 64,
                              'tokenizer_revision': 'tok', 'chat_template_sha256': 'b' * 64,
                              'hidden_size': 3, 'layer_count': 5},
                    'scope': 'full', 'layers': [1, 2], 'alphas': [1.0, 1.0],
                    'directions': {'file': 'directions.f32', 'dtype': 'float32-le', 'byte_count': len(payload),
                                   'sha256': hashlib.sha256(payload).hexdigest()},
                    'provenance': {'extraction_tool': 'fixture', 'extraction_commit': 'fixture',
                                   'dataset_manifests': {'train': '1' * 64, 'validation': '2' * 64,
                                                         'final': '3' * 64}}}
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
            self.assertEqual(release.validate_directory(out, manifest['model']), manifest)
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

    def test_directory_validation_rejects_identity_and_non_unit_rows(self):
        with tempfile.TemporaryDirectory() as d:
            manifest, _ = self.fixture(d)
            with self.assertRaisesRegex(release.ProfileError, 'model identity mismatch'):
                release.validate_directory(d, dict(manifest['model'], revision='other'))
            with open(os.path.join(d, 'directions.f32'), 'wb') as f:
                f.write(struct.pack('<6f', 1, 1, 0, 0, 1, 0))
            manifest['directions']['sha256'] = hashlib.sha256(struct.pack('<6f', 1, 1, 0, 0, 1, 0)).hexdigest()
            with open(os.path.join(d, 'manifest.json'), 'w') as f:
                json.dump(manifest, f)
            with self.assertRaisesRegex(release.ProfileError, 'direction norm'):
                release.validate_directory(d)


if __name__ == '__main__': unittest.main()
