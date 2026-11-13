"""Loss-only adapter for GPT-2 IOI logits at the official answer position."""

import torch


def margin(logits, labels):
    scores = logits.gather(-1, labels)
    return scores[..., 0] - scores[..., 1]


def objective(name, logits, labels, targets, denominator):
    from matryoshka_attribution.losses import attribution_loss

    if name == "margin":
        return ((margin(logits.float(), labels) - targets["margin"].detach()) / denominator).square().mean()
    if name == "kl":
        return attribution_loss("kl", logits.float(), labels[:, 0], labels[:, 1],
                                corrupt_topk=False, clean_logp=targets["logp"].detach())
    if name == "repo_cmd":
        return attribution_loss("cmd", logits.float(), labels[:, 0], labels[:, 1],
                                corrupt_topk=False, target_d=targets["margin"].detach(),
                                corrupt_d=targets["empty"].detach())
    raise ValueError(f"Unregistered loss {name}")
