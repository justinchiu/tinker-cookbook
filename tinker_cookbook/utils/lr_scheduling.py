import logging
import math

logger = logging.getLogger(__name__)


def compute_schedule_lr_multiplier(lr_schedule: str, step: int, total_steps: int) -> float:
    """
    What factor to multiply the base LR by due to the LR schedule
    """
    if lr_schedule == "linear":
        return 1 - step / total_steps
    elif lr_schedule == "constant":
        return 1
    elif lr_schedule == "cosine":
        # Cosine decay from 1 to 0
        return 0.5 * (1 + math.cos(math.pi * step / total_steps))
    elif lr_schedule == "cosine_warmup":
        # Cosine with 5% warmup
        warmup_steps = int(0.05 * total_steps)
        if step < warmup_steps:
            # Linear warmup from 0 to 1
            return step / warmup_steps
        else:
            # Cosine decay from 1 to 0
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unknown learning rate schedule: {lr_schedule}")
