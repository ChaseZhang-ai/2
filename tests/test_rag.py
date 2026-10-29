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


