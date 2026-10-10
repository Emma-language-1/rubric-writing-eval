import unittest

from src.data_prep import build_preference_records, build_sft_records


class DataPreparationTests(unittest.TestCase):
    def test_sft_record_preserves_human_score(self):
        essay = {"prompt": "p", "text": "t", "human_scores": {"task_response": 7}}
        record = build_sft_records([essay], {"traits": {}})[0]
        self.assertEqual(record["output"]["scores"]["task_response"], 7)

    def test_preference_record_has_pair(self):
        item = {"rubric_trait": "evidence", "chosen": "a", "rejected": "b", "essay_id": "1", "source": "synthetic"}
        record = build_preference_records([item])[0]
        self.assertEqual((record["chosen"], record["rejected"]), ("a", "b"))


if __name__ == "__main__":
    unittest.main()
