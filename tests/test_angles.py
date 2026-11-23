import unittest

from zhulu.generation import (
    ANGLE_MODIFIERS,
    ANGLE_TEMPLATES,
    capacity,
    plan_angles,
    prompt_diversity,
)


class TestAnglePlanner(unittest.TestCase):
    def test_returns_requested_count(self):
        self.assertEqual(len(plan_angles("ceramic mug", 6)), 6)

    def test_deterministic_for_same_seed(self):
        a = plan_angles("ceramic mug", 5, seed=11)
        b = plan_angles("ceramic mug", 5, seed=11)
        self.assertEqual([x.key for x in a], [x.key for x in b])

    def test_different_seed_changes_order(self):
        a = plan_angles("ceramic mug", 5, seed=1)
        b = plan_angles("ceramic mug", 5, seed=2)
        self.assertNotEqual([x.key for x in a], [x.key for x in b])

    def test_prompts_are_unique(self):
        angles = plan_angles("ceramic mug", capacity())
        self.assertEqual(len({a.prompt for a in angles}), len(angles))
        self.assertEqual(len({a.key for a in angles}), len(angles))

    def test_one_per_template_within_template_count(self):
        angles = plan_angles("boots", len(ANGLE_TEMPLATES))
        self.assertEqual(len({a.name for a in angles}), len(ANGLE_TEMPLATES))

    def test_avoid_moves_angles_to_back(self):
        avoid = list(ANGLE_TEMPLATES)[:3]
        angles = plan_angles("boots", 4, avoid=avoid)
        self.assertEqual(len({a.name for a in angles}), 4)
        self.assertTrue(all(a.name not in avoid for a in angles[: len(ANGLE_TEMPLATES) - 3]))

    def test_capacity_overflow_raises(self):
        with self.assertRaises(ValueError):
            plan_angles("boots", capacity() + 1)

    def test_empty_product_raises(self):
        with self.assertRaises(ValueError):
            plan_angles("   ", 3)

    def test_diversity_is_high_within_templates(self):
        angles = plan_angles("ceramic mug", len(ANGLE_TEMPLATES))
        self.assertGreater(prompt_diversity(angles), 0.4)

    def test_product_substituted_into_every_prompt(self):
        for a in plan_angles("ceramic mug", len(ANGLE_TEMPLATES)):
            self.assertIn("ceramic mug", a.prompt)


if __name__ == "__main__":
    unittest.main()
