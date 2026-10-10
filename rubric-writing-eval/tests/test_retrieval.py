import unittest

from src.retrieval import RubricRetriever


class RetrievalTests(unittest.TestCase):
    def test_relevant_descriptor_ranks_first(self):
        passages = {
            "evidence": "Uses relevant reasons and examples as evidence.",
            "language": "Uses vocabulary and grammar with control.",
        }
        result = RubricRetriever(passages).retrieve("relevant example evidence", top_k=1)
        self.assertEqual(result[0][0], "evidence")


if __name__ == "__main__":
    unittest.main()
