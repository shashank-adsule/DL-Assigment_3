"""
lr_scheduler.py
---------------
Noam learning-rate scheduler from "Attention Is All You Need" (section 5.3):

    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

Usage:
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0,
                                 betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=256, warmup_steps=4000)

    # inside training loop, call ONCE per gradient update:
    scheduler.step()
"""

import torch


class NoamScheduler:
    """
    Implements the Noam learning-rate schedule.

    The Adam optimiser's base lr should be set to 1.0; the scheduler
    overwrites param_group["lr"] every step with the exact value from
    the formula.

    Args:
        optimizer    : torch.optim.Optimizer
        d_model      : model dimension (same as Transformer d_model)
        warmup_steps : number of linear warm-up steps (paper: 4000)
        factor       : overall scale multiplier (default 1.0)
    """

    def __init__(self, optimizer: torch.optim.Optimizer,
                 d_model: int, warmup_steps: int = 4000,
                 factor: float = 1.0):
        self.optimizer     = optimizer
        self.d_model       = d_model
        self.warmup_steps  = warmup_steps
        self.factor        = factor
        self._step         = 0          # counts gradient updates

    # ── Core formula ──────────────────────────────────────────────────────────

    def _get_lr(self, step: int) -> float:
        """
        Compute the learning rate at a given step.

        During warm-up  (step ≤ warmup_steps) : lr increases linearly.
        After warm-up   (step > warmup_steps)  : lr decays ∝ step^(-0.5).
        """
        step = max(step, 1)   # guard against division by zero
        return self.factor * (
            self.d_model ** (-0.5)
            * min(step ** (-0.5),
                  step * self.warmup_steps ** (-1.5))
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def step(self):
        """Advance the step counter and update the optimizer's lr."""
        self._step += 1
        lr = self._get_lr(self._step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    @property
    def current_lr(self) -> float:
        """Current learning rate (read-only)."""
        return self._get_lr(self._step)

    @property
    def current_step(self) -> int:
        return self._step


# ── Factory function (convenience) ────────────────────────────────────────────

def get_optimizer_and_scheduler(model, d_model: int,
                                 warmup_steps: int = 4000,
                                 factor: float = 1.0):
    """
    Build the Adam optimiser + Noam scheduler pair recommended in the paper.

    Returns:
        optimizer : torch.optim.Adam  (base lr = 1.0)
        scheduler : NoamScheduler
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,              # overridden by scheduler every step
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = NoamScheduler(optimizer, d_model=d_model,
                               warmup_steps=warmup_steps, factor=factor)
    return optimizer, scheduler
