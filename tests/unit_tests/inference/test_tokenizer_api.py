import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "megatron/core/inference/text_generation_server/tokenizer_api.py"
)
SPEC = importlib.util.spec_from_file_location("tokenizer_api", MODULE_PATH)
tokenizer_api = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tokenizer_api)

detokenize_for_api = tokenizer_api.detokenize_for_api
get_tokenizer_info = tokenizer_api.get_tokenizer_info
tokenize_for_api = tokenizer_api.tokenize_for_api


class FakeTokenizer:
    bos_id = 1
    eos_id = 2
    pad_id = 0
    vocab_size = 8
    chat_template = "{prompt}"

    def tokenize(self, prompt):
        if prompt == "already-bos":
            return [self.bos_id, 5]
        return [5]

    def detokenize(self, token_ids, skip_special_tokens=True):
        if skip_special_tokens:
            token_ids = [token_id for token_id in token_ids if token_id not in {0, 1, 2}]
        return " ".join(f"<{token_id}>" for token_id in token_ids)


class PieceBackedSpecialTokenizer(FakeTokenizer):
    def detokenize(self, token_ids, skip_special_tokens=True):
        token_ids = [token_id for token_id in token_ids if token_id not in {0, 1, 2}]
        return " ".join(f"<{token_id}>" for token_id in token_ids)

    def ids_to_tokens(self, token_ids):
        mapping = {0: "<pad>", 1: "<s>", 2: "</s>"}
        return [mapping[token_id] for token_id in token_ids]


class TestTokenizerApi(unittest.TestCase):
    def test_tokenizer_info_preserves_special_tokens(self):
        info = get_tokenizer_info(FakeTokenizer())

        self.assertEqual(
            info,
            {
                "eos_token": "<2>",
                "bos_token": "<1>",
                "pad_token": "<0>",
                "chat_template": "{prompt}",
                "vocab_size": 8,
            },
        )

    def test_tokenizer_info_uses_special_pieces_when_detokenization_is_empty(self):
        info = get_tokenizer_info(PieceBackedSpecialTokenizer())

        self.assertEqual(info["bos_token"], "<s>")
        self.assertEqual(info["eos_token"], "</s>")
        self.assertEqual(info["pad_token"], "<pad>")

    def test_tokenize_for_api_adds_at_most_one_bos(self):
        tokenizer = FakeTokenizer()

        self.assertEqual(tokenize_for_api(tokenizer, "plain"), [5])
        self.assertEqual(tokenize_for_api(tokenizer, "plain", add_special_tokens=True), [1, 5])
        self.assertEqual(
            tokenize_for_api(tokenizer, "already-bos", add_special_tokens=True), [1, 5]
        )

    def test_detokenize_for_api_validates_token_ids_and_keeps_special_tokens(self):
        self.assertEqual(detokenize_for_api(FakeTokenizer(), [1, 5, 2]), "<1> <5> <2>")

        with self.assertRaisesRegex(TypeError, "tokens must be a list of integers"):
            detokenize_for_api(FakeTokenizer(), ["1"])

    def test_detokenize_for_api_uses_special_pieces_when_special_detok_is_empty(self):
        self.assertEqual(detokenize_for_api(PieceBackedSpecialTokenizer(), [1, 5, 2]), "<s><5></s>")


if __name__ == "__main__":
    unittest.main()
