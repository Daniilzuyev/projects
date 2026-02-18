from __future__ import annotations

from src.models.training.configs import config_lgbm_v1_1
from src.models.training.core import train_binary


def main() -> None:
    train_binary(config_lgbm_v1_1())


if __name__ == "__main__":
    main()
