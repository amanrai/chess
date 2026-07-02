import unittest

from chessgm.tokenizer import ChessTokenizer, TOKEN_TO_ID, VOCAB, tokenize_san


class TokenizerTests(unittest.TestCase):
    def test_pad_is_explicit_zero(self):
        self.assertEqual(VOCAB[0], "<PAD>")
        self.assertEqual(TOKEN_TO_ID["<PAD>"], 0)

    def test_move_round_trips_through_ids(self):
        tok = ChessTokenizer()
        examples = [
            "e4",
            "Nbd7",
            "exd8=Q+",
            "O-O",
            "O-O-O#",
            "R1e2",
            "Qh5#",
            "Bxc6",
            "dxc6",
        ]
        for san in examples:
            with self.subTest(san=san):
                encoded = tok.encode_move(san)
                self.assertEqual(encoded.tokens, tokenize_san(san))
                self.assertEqual(tok.decode_ids(encoded.ids), encoded.tokens)
                self.assertEqual(tok.decode_move(encoded.ids), san)

    def test_results_round_trip_without_eom(self):
        tok = ChessTokenizer()
        for result in ["1-0", "0-1", "1/2-1/2", "*"]:
            with self.subTest(result=result):
                encoded = tok.encode_move(result)
                self.assertNotIn("<EOM>", encoded.tokens)
                self.assertEqual(tok.decode_move(encoded.ids), result)

    def test_movetext_round_trip_canonical(self):
        tok = ChessTokenizer()
        movetext = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Bxc6 dxc6 5. O-O Nf6 1/2-1/2"
        ids = tok.encode_movetext(movetext)
        restored = tok.detokenize_movetext(ids)
        self.assertEqual(restored, movetext)


if __name__ == "__main__":
    unittest.main()
