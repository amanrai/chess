import tempfile
import unittest
from pathlib import Path

import torch

from chessgm.network_q import QFormerPlyHistoryEncoder, QInverseTransitionDecoder


class QInverseTransitionDecoderTests(unittest.TestCase):
    def setUp(self):
        self.kwargs = dict(
            vocab_size=32,
            ply_expr=8,
            model_dim=16,
            heads=4,
            history_layers=1,
            q_layers=1,
            num_queries=4,
            transition_layers=1,
        )

    def test_decodes_unpooled_state_transition_to_packet_logits(self):
        model = QInverseTransitionDecoder(**self.kwargs)
        before = torch.randint(0, 32, (2, 3, 8))
        after = torch.randint(0, 32, (2, 3, 8))

        logits = model(before, after)

        self.assertEqual(logits.shape, (2, 8, 32))
        logits.sum().backward()
        self.assertIsNotNone(model.encoder.query_tokens.grad)
        self.assertIsNotNone(model.inverse_attention.cross_attn.in_proj_weight.grad)

    def test_loads_and_freezes_only_prefixed_encoder_weights(self):
        source = QFormerPlyHistoryEncoder(
            vocab_size=32,
            ply_expr=8,
            model_dim=16,
            heads=4,
            history_layers=1,
            q_layers=1,
            num_queries=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "probe.pt"
            torch.save(
                {
                    "model": {
                        **{f"encoder.{key}": value for key, value in source.state_dict().items()},
                        "check_head.weight": torch.randn(2, 16),
                    }
                },
                checkpoint_path,
            )
            model = QInverseTransitionDecoder(
                **self.kwargs,
                pretrained_encoder_checkpoint=checkpoint_path,
                freeze_encoder=True,
            )

        self.assertEqual(model.loaded_encoder_tensors, len(source.state_dict()))
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.encoder.parameters())
        )
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, model.encoder.state_dict()[key]), key)
        self.assertTrue(any(parameter.requires_grad for parameter in model.head.parameters()))


if __name__ == "__main__":
    unittest.main()
