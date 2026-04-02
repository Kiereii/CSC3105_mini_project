import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


STEP_SCRIPTS = {
    "preprocess": "src/shared/preprocessing/preprocess_data.py",
    "second_path": "src/shared/preprocessing/second_path_features.py",
    "dt_exploratory": "src/experimental/dt_exploratory.py",
    "random_forest": "src/experimental/random_forest_classifier.py",
    "logreg_svm": "src/experimental/logreg_svm_classifier.py",
    "xgboost": "src/experimental/xgboost_classifier.py",
    "compare": "src/experimental/compare_models.py",
    "pair_classifier": "src/task1/xgboost_pair_classifier.py",
    "range_regressor": "src/experimental/range_regressor.py",
    "xgboost_range_regressor": "src/task2/xgboost_range_regressor.py",
}

STEP_ALIASES = {
    "xgb_range_regressor": "xgboost_range_regressor",
}

DEFAULT_STEP_ORDER = [
    "preprocess",
    "second_path",
    "dt_exploratory",
    "random_forest",
    "logreg_svm",
    "xgboost",
    "compare",
    "pair_classifier",
    "range_regressor",
    "xgboost_range_regressor",
]


def normalize_step_name(step: str) -> str:
    return STEP_ALIASES.get(step, step)


def parse_step_list(raw: str) -> set[str]:
    return {
        normalize_step_name(item.strip()) for item in raw.split(",") if item.strip()
    }


def validate_steps(steps: set[str], arg_name: str) -> None:
    invalid = sorted(s for s in steps if s not in STEP_SCRIPTS)
    if invalid:
        raise ValueError(
            f"Invalid value(s) for {arg_name}: {', '.join(invalid)}. "
            f"Allowed: {', '.join(DEFAULT_STEP_ORDER)}"
        )


def run_step(script_path: str, env: dict) -> None:
    print(f"\n>>> Running: {script_path}")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script_path)],
        check=True,
        env=env,
        cwd=PROJECT_ROOT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UWB experiment pipeline")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=(
            "Comma-separated steps to run. "
            "Example: --only preprocess,second_path,random_forest"
        ),
    )
    parser.add_argument(
        "--skip",
        type=str,
        default=None,
        help=("Comma-separated steps to skip. Example: --skip xgboost,compare"),
    )
    args = parser.parse_args()

    if (
        args.val_size <= 0
        or args.test_size <= 0
        or (args.val_size + args.test_size) >= 1
    ):
        raise ValueError(
            "--val-size and --test-size must be > 0 and their sum must be < 1."
        )

    train_size = 1 - args.val_size - args.test_size

    env = os.environ.copy()
    env["VAL_SIZE"] = str(args.val_size)
    env["TEST_SIZE"] = str(args.test_size)
    env["RANDOM_SEED"] = str(args.seed)

    only_steps = parse_step_list(args.only) if args.only else set()
    skip_steps = parse_step_list(args.skip) if args.skip else set()

    validate_steps(only_steps, "--only")
    validate_steps(skip_steps, "--skip")

    if only_steps:
        step_order = [
            s for s in DEFAULT_STEP_ORDER if s in only_steps and s not in skip_steps
        ]
    else:
        step_order = [s for s in DEFAULT_STEP_ORDER if s not in skip_steps]

    if not step_order:
        raise ValueError("No steps selected to run. Check --only/--skip values.")

    print("=" * 80)
    print("RUN CONFIG")
    print("=" * 80)
    print(f"SPLIT       : {train_size:.2f}/{args.val_size:.2f}/{args.test_size:.2f}")
    print(f"VAL_SIZE    : {args.val_size}")
    print(f"TEST_SIZE   : {args.test_size}")
    print(f"RANDOM_SEED : {args.seed}")
    print(f"STEPS       : {', '.join(step_order)}")
    print("=" * 80)

    for step in step_order:
        run_step(STEP_SCRIPTS[step], env)

    print("\nPipeline complete.")
    print("Preprocessed data: outputs/preprocessed/")
    print("Model outputs:     outputs/")


if __name__ == "__main__":
    main()
