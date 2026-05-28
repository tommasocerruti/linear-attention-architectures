import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[3] / "tools/run_loglikelihood_scoring_server.py"
SPEC = importlib.util.spec_from_file_location("run_loglikelihood_scoring_server", MODULE_PATH)
scoring_server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scoring_server)

build_lm_eval_completion_choice = scoring_server.build_lm_eval_completion_choice


class TestLmEvalScoringResponse(unittest.TestCase):
    def test_lm_eval_slice_scores_only_continuation_tokens(self):
        choice = build_lm_eval_completion_choice(
            index=0,
            text="The capital of France is Paris",
            token_texts=["The", " capital", " of", " France", " is", " Paris"],
            text_offsets=[0, 3, 11, 14, 21, 24],
            next_token_logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            next_token_top_logprobs=[
                {" capital": -0.1},
                {" of": -0.2},
                {" France": -0.3},
                {" is": -0.4},
                {" Paris": -0.5},
            ],
        )

        logprobs = choice["logprobs"]
        self.assertEqual(len(logprobs["tokens"]), 7)
        self.assertEqual(len(logprobs["token_logprobs"]), 7)
        self.assertEqual(len(logprobs["top_logprobs"]), 7)
        self.assertEqual(len(logprobs["text_offset"]), 7)

        ctxlen = 5
        self.assertEqual(logprobs["tokens"][ctxlen:-1], [" Paris"])
        self.assertEqual(logprobs["token_logprobs"][ctxlen:-1], [-0.5])
        self.assertEqual(logprobs["top_logprobs"][ctxlen:-1], [{" Paris": -0.5}])

    def test_validates_response_lengths(self):
        with self.assertRaisesRegex(ValueError, "len\\(token_texts\\) - 1"):
            build_lm_eval_completion_choice(
                index=0,
                text="a b",
                token_texts=["a", " b"],
                text_offsets=[0, 1],
                next_token_logprobs=[],
                next_token_top_logprobs=None,
            )


if __name__ == "__main__":
    unittest.main()
