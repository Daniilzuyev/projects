from __future__ import annotations

from src.models.training.configs import config_lgbm_rank_v1
from src.models.training.core import train_rank


def main() -> None:
    train_rank(config_lgbm_rank_v1())


if __name__ == "__main__":
    main()
