"""Example training loop for the bilingual reasoner."""
from __future__ import annotations

import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from iqgp.data.datasets import SyntheticBilingualDataset, collate_examples
from iqgp.models.backbone import BilingualReasoner
from iqgp.objectives.lexicon import LexiconProjector
from iqgp.objectives.losses import (
    code_switch_consistency,
    emd_bilingual,
    entity_unit_agreement,
    info_nce_loss,
    language_eraser_loss,
)
from iqgp.objectives.adversary import LanguageAdversary


def build_candidates(batch_size: int, max_nodes: int, num_entities: int, num_units: int, hidden_size: int) -> Tuple[tuple, tuple]:
    ent_embeds = torch.randn(batch_size, num_entities, hidden_size)
    ent_mask = torch.ones(batch_size, num_entities, dtype=torch.bool)
    unit_embeds = torch.randn(batch_size, num_units, hidden_size)
    unit_mask = torch.ones(batch_size, num_units, dtype=torch.bool)
    return (ent_embeds, ent_mask), (unit_embeds, unit_mask)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticBilingualDataset(
        length=args.steps,
        seq_len=args.seq_len,
        hidden_size=args.hidden,
        vocab_size=args.vocab,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_examples)

    model = BilingualReasoner(
        hidden_size=args.hidden,
        vocab_en=args.vocab,
        vocab_zh=args.vocab,
        max_nodes=args.max_nodes,
        codebook_size=args.codebook,
        num_entities=args.entities,
        num_units=args.units,
    ).to(device)
    projector = LexiconProjector(args.vocab, args.lexicon).to(device)
    adversary = LanguageAdversary(args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step, batch in enumerate(loader):
        en_tokens, zh_tokens, en_mask, zh_mask, answer_en, answer_zh = batch
        en_tokens = en_tokens.to(device)
        zh_tokens = zh_tokens.to(device)
        en_mask = en_mask.to(device)
        zh_mask = zh_mask.to(device)
        answer_en = answer_en.to(device)
        answer_zh = answer_zh.to(device)

        entity_candidates, unit_candidates = build_candidates(en_tokens.size(0), args.max_nodes, args.entities, args.units, args.hidden)
        entity_candidates = (entity_candidates[0].to(device), entity_candidates[1].to(device))
        unit_candidates = (unit_candidates[0].to(device), unit_candidates[1].to(device))

        output = model(en_tokens, zh_tokens, entity_candidates, unit_candidates, en_mask=en_mask, zh_mask=zh_mask)

        ce_en = F.cross_entropy(output.en_logits.view(-1, output.en_logits.size(-1)), answer_en.view(-1))
        ce_zh = F.cross_entropy(output.zh_logits.view(-1, output.zh_logits.size(-1)), answer_zh.view(-1))
        task_loss = ce_en + ce_zh

        emd_loss = emd_bilingual(output.en_logits, output.zh_logits, projector)
        pooled_en = output.graph.node_states.mean(dim=1)
        pooled_zh = output.graph.node_states.mean(dim=1)
        plan_loss = info_nce_loss(pooled_en, pooled_zh)

        ent_loss = entity_unit_agreement(output.graph.qid_logits, output.graph.qid_logits, output.graph.unit_logits, output.graph.unit_logits)
        csd_loss = code_switch_consistency(output.en_logits.view(-1, output.en_logits.size(-1)), answer_en.view(-1))

        adversary_logits = adversary(output.graph.node_states.view(-1, output.graph.node_states.size(-1)))
        lang_labels = torch.zeros(adversary_logits.size(0), dtype=torch.long, device=device)
        erase_loss = language_eraser_loss(adversary_logits, lang_labels)

        loss = task_loss + 0.5 * emd_loss + 0.5 * plan_loss + 0.5 * ent_loss + 0.2 * csd_loss + 0.2 * erase_loss + output.vq_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0:
            print(f"step={step} loss={loss.item():.4f}")

    print("Training complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the I-QGP bilingual reasoner on synthetic data")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--max-nodes", type=int, default=6)
    parser.add_argument("--codebook", type=int, default=16)
    parser.add_argument("--entities", type=int, default=10)
    parser.add_argument("--units", type=int, default=6)
    parser.add_argument("--lexicon", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
