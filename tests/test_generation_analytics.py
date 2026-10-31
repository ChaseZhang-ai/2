import unittest

from zhulu.analytics import attribute, campaign_roas, classify_sentiment, summarize_voc
from zhulu.analytics.roi import AssetPerformance, Event
from zhulu.generation import (
    ModelImageRequest,
    VideoRequest,
    build_model_image_graph,
    build_product_video_graph,
    denoise_schedule,
    frame_interpolation_plan,
    lip_sync_gate,
)


class TestGenerationGraph(unittest.TestCase):
    def test_model_image_graph_is_dag(self):
        g = build_model_image_graph(ModelImageRequest(product_image="p.png", seed=7))
        order = g.topo_order()
        self.assertEqual(len(order), len(g))
        self.assertLess(order.index("1"), order.index("11"))

    def test_signature_ignores_seed_and_prompt(self):
        a = build_model_image_graph(ModelImageRequest(product_image="p.png", seed=1, prompt="red"))
        b = build_model_image_graph(ModelImageRequest(product_image="p.png", seed=2, prompt="blue"))
        self.assertEqual(a.signature(), b.signature())

    def test_signature_changes_with_model(self):
        a = build_model_image_graph(ModelImageRequest(product_image="p.png"))
        b = build_model_image_graph(ModelImageRequest(product_image="p.png", scene_lora="other.safetensors"))
        self.assertNotEqual(a.signature(), b.signature())

    def test_comfy_round_trip(self):
        g = build_product_video_graph(VideoRequest(product_image="p.png"))
        payload = g.to_comfy()
        g2 = type(g).from_comfy(payload)
        self.assertEqual(g.signature(), g2.signature())

    def test_denoise_schedule_shape(self):
        s = denoise_schedule(10)
        self.assertEqual(len(s), 10)
        self.assertGreater(s[0], s[-1])
        self.assertTrue(all(0 < v <= 1 for v in s))

    def test_frame_interpolation_plan(self):
        pairs = frame_interpolation_plan(240, keyframe_fps=6, fps=24)
        self.assertTrue(all(b - a == 1 for a, b in pairs))

    def test_lip_sync_gate(self):
        good = lip_sync_gate(lse_c=0.9, lse_d=0.1)
        self.assertTrue(good["accept"])
        bad = lip_sync_gate(lse_c=0.3, lse_d=0.8)
        self.assertTrue(bad["rerun_mouth"])


class TestAnalytics(unittest.TestCase):
    def test_roi_and_roas(self):
        events = [
            Event("v1", "impression", ts=1),
            Event("v1", "impression", ts=2),
            Event("v1", "click", ts=3),
            Event("v1", "conversion", value=100.0, ts=4),
            Event("v2", "impression", ts=1),
            Event("v2", "click", ts=2),
        ]
        perf = attribute(events, spend={"v1": 10.0, "v2": 5.0})
        self.assertEqual(perf["v1"].conversions, 1)
        self.assertAlmostEqual(perf["v1"].roas, 10.0)
        agg = campaign_roas(perf)
        self.assertAlmostEqual(agg["revenue"], 100.0)
        self.assertAlmostEqual(agg["roas"], 100.0 / 15.0)

    def test_sentiment_polarity(self):
        self.assertGreater(classify_sentiment("质量很好，物流也快，推荐"), 0)
        self.assertLess(classify_sentiment("质量很差，物流太慢，失望"), 0)
        self.assertGreater(classify_sentiment("质量不错，做工精细"), 0)
        # Negation flips a positive term.
        self.assertLess(classify_sentiment("质量不好"), 0)

    def test_voc_aspects(self):
        reviews = [
            "质量很好，做工精细",
            "质量太差了，有破损",
            "物流很快，第二天就到",
            "价格实惠，值得推荐",
        ]
        summaries = summarize_voc(reviews)
        aspects = {s.aspect: s for s in summaries}
        self.assertIn("quality", aspects)
        self.assertIn("shipping", aspects)
        self.assertGreater(aspects["shipping"].mean_sentiment, 0.5)


if __name__ == "__main__":
    unittest.main()
