import math
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if 'aiohttp' not in sys.modules:  # sampling itself is stdlib-only; keep this unit test usable without frontend extras
    sys.modules['aiohttp'] = types.SimpleNamespace(web=types.SimpleNamespace(Response=object, json_response=lambda *a, **k: None))
from g6front import sampling  # noqa: E402
from g6front.errors import ApiError  # noqa: E402


class SteeringTests(unittest.TestCase):
    cap = {'profile': 'example', 'digest': '0' * 64, 'scope': 'full', 'layers': 48,
           'forced': False, 'forced_scale': 0.0}

    def test_off_and_canonical_zero(self):
        self.assertEqual(sampling.steering({}, self.cap), {})
        self.assertEqual(sampling.steering({'steering_scale': -0.0}, self.cap), {})

    def test_profile_default_and_fp32_scale(self):
        self.assertEqual(sampling.steering({'steering_profile': 'example'}, self.cap),
                         {'steering_profile': 'example', 'steering_scale': 1.0})
        got = sampling.steering({'steering_profile': 'example', 'steering_scale': 1.1}, self.cap)
        self.assertNotEqual(got['steering_scale'], 1.1)  # canonical IEEE-754 binary32 value

    def test_rejections(self):
        cases = [
            ({'steering_profile': 'other'}, 'steering_profile'),
            ({'steering_profile': 7}, 'steering_profile'),
            ({'steering_profile': 'bad/name'}, 'steering_profile'),
            ({'steering_scale': 1}, 'steering_scale'),
            ({'steering_profile': 'example', 'steering_scale': math.inf}, 'steering_scale'),
            ({'steering_profile': 'example', 'steering_scale': 4.1}, 'steering_scale'),
        ]
        for body, param in cases:
            with self.subTest(body=body):
                with self.assertRaises(ApiError) as e:
                    sampling.steering(body, self.cap)
                self.assertEqual(e.exception.param, param)

    def test_forced_conflict(self):
        cap = dict(self.cap, forced=True, forced_scale=2.0)
        self.assertEqual(sampling.steering({}, cap), {})
        with self.assertRaises(ApiError):
            sampling.steering({'steering_profile': 'example'}, cap)
        self.assertEqual(sampling.steering({'steering_profile': 'example', 'steering_scale': 2}, cap),
                         {'steering_profile': 'example', 'steering_scale': 2.0})

    def test_profile_on_a_server_without_steering_is_clear(self):
        for cap in (None, {}):
            with self.assertRaises(ApiError) as e:
                sampling.steering({'steering_profile': 'example'}, cap)
            self.assertEqual(e.exception.param, 'steering_profile')
            self.assertEqual(str(e.exception),
                             'steering is not enabled on this server (start it with --steering-profile)')


if __name__ == '__main__':
    unittest.main()
