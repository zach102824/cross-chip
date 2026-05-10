from __future__ import annotations

import argparse
import sys
from pathlib import Path


def ensure_imports(shadowgrouping_root: Path) -> None:
    root = shadowgrouping_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"shadowgrouping root not found: {root}. "
            "Set --shadowgrouping-root to your local shadowgrouping project."
        )

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from shadowgrouping.measurement_schemes import (  # type: ignore
        AdaptiveShadows,
        Derandomization,
        SettingSampler,
        Shadow_Grouping,
    )
    from shadowgrouping.weight_functions import Bernstein_bound, Inconfidence_bound  # type: ignore

    _ = (
        AdaptiveShadows,
        Derandomization,
        SettingSampler,
        Shadow_Grouping,
        Bernstein_bound,
        Inconfidence_bound,
    )


def test_shadowgrouping_imports() -> None:
    ensure_imports(Path("/Users/zacharyhe/shadowgrouping"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check that shadowgrouping imports are available."
    )
    parser.add_argument(
        "--shadowgrouping-root",
        type=Path,
        default=Path("/Users/zacharyhe/shadowgrouping"),
    )
    args = parser.parse_args()
    ensure_imports(args.shadowgrouping_root)
    print("PASS: shadowgrouping imports succeeded.")


if __name__ == "__main__":
    main()
