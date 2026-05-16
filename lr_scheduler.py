import torch

class NoamScheduler:
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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,              # overridden by scheduler every step
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = NoamScheduler(optimizer, d_model=d_model,
                               warmup_steps=warmup_steps, factor=factor)
    return optimizer, scheduler