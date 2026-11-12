"""Exit profile: how often each exit answers and how accurate it is, for every threshold tau.

Measured on the VALIDATION split only; the test split is for reporting.

A device that runs blocks [0, k) checks the exits before k in order, and the first whose
confidence (largest softmax) is >= tau answers. The exits in use are therefore always the first
m of the model's exits, and the profile gives, for every tau and every m:

    exit_rate[j]      fraction of ALL inputs that exit j answers (the exits before it in use)
    exit_accuracy[j]  accuracy of exit j on the inputs it answers; None if it answers none
    p_continue[j]     fraction of inputs that get past exits 0..j, i.e. are sent on (or finished
                      locally): the P_continue of the latency formula for a k just behind exit j
    accuracy[j]       accuracy of the whole system with exits 0..j in use: the exits' answers
                      where they answer, the final output for the rest

With no exit before k (m = 0) nothing exits: P_continue is 1 and the accuracy is
`final_accuracy`. At tau = 1.0 no exit is taken by definition (models/split.py), so the rates
are 0 there whatever the confidences. `head_accuracy[j]` is the accuracy exit j would have if it
answered every input.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from edgeinf.data.imagenette import normalise
from edgeinf.models.multi_exit import MultiExitModel
from edgeinf.models.split import TAU_NEVER

TAUS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)  # thresholds the scheduler can choose from


@torch.inference_mode()
def exit_outputs(
    model: MultiExitModel, frames: np.ndarray, batch_size: int, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What every exit and the final output say about each 8-bit frame, shape (N, 3, H, W).

    Returns (confidence, predicted, final): float (N, E) and int (N, E) for the model's E exits
    in block order, and the final output's label, int (N,). No exit stops a frame here.
    """
    model = model.to(device).eval()
    exits = model.exit_points
    confidence, predicted, final = [], [], []
    for start in range(0, len(frames), batch_size):
        x = normalise(torch.from_numpy(np.array(frames[start : start + batch_size])))
        x = x.to(device)
        batch_confidence, batch_predicted = [], []
        for block in range(model.num_blocks):
            x = model.blocks[block](x)
            if block in exits:
                top, label = model.exits[str(block)](x).softmax(1).max(1)
                batch_confidence.append(top.cpu())
                batch_predicted.append(label.cpu())
        shape = (x.shape[0], 0)
        confidence.append(torch.stack(batch_confidence, 1) if exits else torch.zeros(shape))
        predicted.append(
            torch.stack(batch_predicted, 1) if exits else torch.zeros(shape, dtype=torch.long)
        )
        final.append(x.argmax(1).cpu())
    return (
        torch.cat(confidence).numpy(),
        torch.cat(predicted).numpy(),
        torch.cat(final).numpy(),
    )


def exit_profile(
    confidence: np.ndarray,
    predicted: np.ndarray,
    final: np.ndarray,
    labels: np.ndarray,
    exit_blocks: Sequence[int],
    taus: Sequence[float] = TAUS,
) -> dict:
    """The exit profile (see the module docstring) from the outputs of `exit_outputs`.

    All rates and accuracies are fractions, 0..1.
    """
    samples, n_exits = confidence.shape
    if n_exits != len(exit_blocks) or predicted.shape != confidence.shape:
        raise ValueError(f"expected one column per exit {list(exit_blocks)}")
    if not samples or len(final) != samples or len(labels) != samples:
        raise ValueError("need the same, non-zero number of inputs in every array")
    labels = np.asarray(labels)
    exit_correct = predicted == labels[:, None]
    final_correct = final == labels

    by_tau = []
    for tau in taus:
        remaining = np.ones(samples, dtype=bool)  # inputs no exit has answered yet
        correct_so_far = 0  # correct answers given by the exits in use
        row = {"tau": tau, "exit_rate": [], "exit_accuracy": [], "p_continue": [], "accuracy": []}
        for j in range(n_exits):
            takes = remaining & (confidence[:, j] >= tau) & (tau < TAU_NEVER)
            taken = int(takes.sum())
            correct = int(exit_correct[takes, j].sum())
            remaining &= ~takes
            correct_so_far += correct
            row["exit_rate"].append(taken / samples)
            row["exit_accuracy"].append(correct / taken if taken else None)
            row["p_continue"].append(float(remaining.mean()))
            row["accuracy"].append((correct_so_far + int(final_correct[remaining].sum())) / samples)
        by_tau.append(row)
    return {
        "exit_blocks": list(exit_blocks),
        "samples": samples,
        "final_accuracy": float(final_correct.mean()),
        "head_accuracy": [float(exit_correct[:, j].mean()) for j in range(n_exits)],
        "by_tau": by_tau,
    }
