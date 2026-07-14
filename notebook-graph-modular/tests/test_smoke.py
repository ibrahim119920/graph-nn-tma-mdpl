import unittest

from scripts.smoke_test import run_smoke_test


class SmokeTest(unittest.TestCase):
    def test_forward_pass(self):
        self.assertEqual(run_smoke_test(), (1, 2))


if __name__ == "__main__":
    unittest.main()
