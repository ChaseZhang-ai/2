import unittest

from zhulu.rag import (
    BM25,
    CustomerServiceAgent,
    Document,
    HybridRetriever,
    detect_language,
    rrf_fuse,
)


class TestBM25(unittest.TestCase):
    def test_ranks_matching_doc_first(self):
        b = BM25()
        b.add("d1", "our warranty covers manufacturing defects for 24 months")
        b.add("d2", "shipping takes 3 to 5 business days within the EU")
        b.add("d3", "you can pay by card or bank transfer")
        top = b.search("how long is the warranty", top_k=1)
        self.assertEqual(top[0][0], "d1")

    def test_cjk_bigram_matching(self):
        b = BM25()
        b.add("z1", "保修期为二十四个月，覆盖制造缺陷")
        b.add("z2", "发货时间通常为三到五个工作日")
        top = b.search("保修期是多久", top_k=1)
        self.assertEqual(top[0][0], "z1")

    def test_empty_query(self):
        b = BM25()
        b.add("d", "hello")
        self.assertEqual(b.search("", top_k=3), [])


class TestRRF(unittest.TestCase):
    def test_fuses_rankings(self):
        fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
        ids = [d for d, _ in fused]
        self.assertEqual(ids[:2], ["a", "b"])
        self.assertIn("d", ids)


def fake_embed(text: str):
    # Deterministic tiny embedding: char-bag hashed into 8 dims.
    v = [0.0] * 8
    for ch in text.lower():
        if ch.isalnum():
            v[ord(ch) % 8] += 1.0
    return v


class TestHybridRetriever(unittest.TestCase):
    def setUp(self):
        self.r = HybridRetriever(embed=fake_embed)
        self.r.add_all([
            Document("k1", "warranty covers defects for 24 months"),
            Document("k2", "shipping takes 3-5 business days"),
            Document("k3", "refunds are processed within 7 days"),
        ])

    def test_hybrid_finds_relevant(self):
        hits = self.r.search("warranty duration", top_k=1)
        self.assertEqual(hits[0][0].doc_id, "k1")

    def test_dense_only_still_returns(self):
        r = HybridRetriever(embed=fake_embed)
        r.add_all([Document("k1", "warranty"), Document("k2", "shipping")])
        hits = r.search("shipping", top_k=1)
        self.assertTrue(hits)


class TestLanguageAndAgent(unittest.TestCase):
    def test_detect_language(self):
        self.assertEqual(detect_language("你好，请问保修期是多久？"), "zh")
        self.assertEqual(detect_language("How long is the warranty?"), "en")
        self.assertEqual(detect_language("保証期間はどのくらいですか？"), "ja")
        self.assertEqual(detect_language("보증 기간은 얼마나 되나요?"), "ko")

    def test_agent_answers_and_escalates(self):
        r = HybridRetriever(embed=fake_embed)
        r.add_all([Document("k1", "warranty covers defects for 24 months")])
        agent = CustomerServiceAgent(retriever=r, min_score=0.001)
        a = agent.answer("warranty")
        self.assertFalse(a.escalated)
        self.assertEqual(a.sources, ["k1"])
        # Unrelated query with a high bar -> escalate.
        strict = CustomerServiceAgent(retriever=r, min_score=10.0)
        self.assertTrue(strict.answer("warranty").escalated)


if __name__ == "__main__":
    unittest.main()
