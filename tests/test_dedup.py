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


class TestBKTree(unittest.TestCase):
    def test_exact_and_radius_search(self):
        tree = BKTree()
        for k in (0b0000, 0b1111, 0b1110):
            tree.add(k)
        assert tree.search(0b1111, 0) == [(0b1111, 0)]
        found = {k for k, _ in tree.search(0b1111, 1)}
        self.assertEqual(found, {0b1111, 0b1110})


class TestNearDuplicateIndex(unittest.TestCase):
    def test_exact_duplicate_detected(self):
        idx = NearDuplicateIndex(radius=6)
        img = gradient()
        idx.add("a", fingerprint(img, 32, 32))
        matches = idx.query(fingerprint(img, 32, 32))
        self.assertTrue(matches)
        self.assertEqual(matches[0].asset_id, "a")
        self.assertEqual(matches[0].distance, 0)

    def test_crop_is_near_duplicate(self):
        idx = NearDuplicateIndex(radius=12)
        full = [((x * 3 + y) % 256) for y in range(24) for x in range(24)]
        idx.add("full", fingerprint(full, 24, 24))
        # Take an inner crop and resize conceptually by re-fingerprinting.
        crop = [((x * 3 + y) % 256) for y in range(4, 22) for x in range(4, 22)]
        fp_crop = fingerprint(crop, 18, 18)
        self.assertTrue(idx.query(fp_crop))


class TestSemanticDedup(unittest.TestCase):
    def test_rejects_similar_accepts_diverse(self):
        s = SemanticDedup(threshold=0.9)
        self.assertTrue(s.accept("a", [1.0, 0.0, 0.0]))
        self.assertFalse(s.accept("b", [0.99, 0.01, 0.0]))
        self.assertTrue(s.accept("c", [0.0, 1.0, 0.0]))

    def test_select_returns_diverse_subset(self):
        s = SemanticDedup(threshold=0.95)
        kept = s.select([
            ("a", [1.0, 0.0]),
            ("b", [0.999, 0.001]),
            ("c", [0.0, 1.0]),
        ])
        self.assertEqual(kept, ["a", "c"])


if __name__ == "__main__":
    unittest.main()
