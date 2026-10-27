import unittest

from zhulu.dedup import (
    BKTree,
    NearDuplicateIndex,
    SemanticDedup,
    average_hash,
    difference_hash,
    fingerprint,
    hamming,
    phash,
)


def gradient(w=32, h=32, shift=0):
    return [((x * 4 + shift) % 256) for y in range(h) for x in range(w)]


class TestHashes(unittest.TestCase):
    def test_identical_images_hash_equal(self):
        img = gradient()
        self.assertEqual(average_hash(img, 32, 32), average_hash(img, 32, 32))
        self.assertEqual(phash(img, 32, 32), phash(img, 32, 32))

    def test_hamming_self_zero(self):
        self.assertEqual(hamming(0xDEADBEEF, 0xDEADBEEF), 0)

    def test_dhash_brightness_shift_invariant(self):
        # A global additive brightness change should not move dhash much.
        base = [((x * 5) % 200) for y in range(16) for x in range(16)]
        brighter = [min(255, p + 20) for p in base]
        d_base = difference_hash(base, 16, 16)
        d_shift = difference_hash(brighter, 16, 16)
        self.assertLessEqual(hamming(d_base, d_shift), 12)

    def test_phash_64bit_range(self):
        img = gradient()
        self.assertLess(phash(img, 32, 32), 1 << 64)


