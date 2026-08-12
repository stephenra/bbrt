"""``bbrt`` command-line interface.

Subcommands:
  * ``bbrt process``    -- SMILES pairs -> tokenized-SELFIES corpus + vocab
  * ``bbrt train``      -- train the Transformer seq2seq
  * ``bbrt generate``   -- run the BBRT recursive-translation optimization loop
  * ``bbrt drd2-fetch`` -- download the public DRD2 activity dataset
  * ``bbrt drd2-train`` -- train the DRD2 Transformer-encoder classifier
"""

from __future__ import annotations

import argparse

from bbrt._logging import configure_logging
from bbrt.config import BBRTConfig, DRD2Config, TrainConfig, load_config


def _cmd_process(args: argparse.Namespace) -> None:
    from bbrt.data.process import process

    process(args.pairs, args.out, valid_frac=args.valid_frac, min_freq=args.min_freq)


def _cmd_train(args: argparse.Namespace) -> None:
    from bbrt.train import train

    cfg = load_config(args.config, TrainConfig) if args.config else TrainConfig()
    if args.data_dir:
        cfg.data.data_dir = args.data_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.max_steps:
        cfg.optim.max_steps = args.max_steps
    if args.precision:
        cfg.precision = args.precision
    train(cfg)


def _cmd_generate(args: argparse.Namespace) -> None:
    from bbrt.generate import run_bbrt

    cfg = load_config(args.config, BBRTConfig) if args.config else BBRTConfig()
    for field in (
        "checkpoint",
        "vocab",
        "seed_file",
        "output_dir",
        "score_func",
        "translate_type",
        "device",
        "num_iters",
    ):
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)
    run_bbrt(cfg)


def _cmd_drd2_fetch(args: argparse.Namespace) -> None:
    from bbrt.scoring.drd2_fetch import fetch

    fetch(out_dir=args.out, force=args.force)


def _cmd_drd2_train(args: argparse.Namespace) -> None:
    from bbrt.train_drd2 import train_drd2

    cfg = load_config(args.config, DRD2Config) if args.config else DRD2Config()
    if args.data:
        cfg.train_csv = args.data
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.max_rows is not None:
        cfg.max_rows = args.max_rows
    if args.init_from:
        cfg.init_from = args.init_from
    if args.precision:
        cfg.precision = args.precision
    train_drd2(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bbrt", description=__doc__)
    parser.add_argument(
        "--log-level", default="INFO", help="logging level (DEBUG/INFO/WARNING/ERROR)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="SMILES pairs -> SELFIES corpus + vocab")
    p.add_argument("--pairs", required=True, help="two-column SMILES pairs file")
    p.add_argument("--out", required=True, help="output data directory")
    p.add_argument("--valid-frac", type=float, default=0.1)
    p.add_argument("--min-freq", type=int, default=1)
    p.set_defaults(func=_cmd_process)

    p = sub.add_parser("train", help="train the Transformer seq2seq")
    p.add_argument("--config", help="path to a train YAML config")
    p.add_argument("--data-dir")
    p.add_argument("--output-dir")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--precision")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("generate", help="run the BBRT optimization loop")
    p.add_argument("--config", help="path to a bbrt YAML config")
    p.add_argument("--checkpoint")
    p.add_argument("--vocab")
    p.add_argument("--seed-file", dest="seed_file")
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--score-func", dest="score_func", choices=["logp04", "qed", "drd2"])
    p.add_argument("--translate-type", dest="translate_type", choices=["sd", "beam"])
    p.add_argument("--num-iters", dest="num_iters", type=int)
    p.add_argument("--device", choices=["cuda", "cpu", "mps"])
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("drd2-fetch", help="download the public DRD2 dataset")
    p.add_argument("--out", default="data/drd2", help="output data directory")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    p.set_defaults(func=_cmd_drd2_fetch)

    p = sub.add_parser("drd2-train", help="train the DRD2 classifier")
    p.add_argument("--config", help="path to a drd2 YAML config")
    p.add_argument("--data", help="labeled CSV override (SMILES + activity)")
    p.add_argument("--data-dir")
    p.add_argument("--output-dir")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--max-rows", type=int, help="subsample cap for quick runs")
    p.add_argument("--init-from", help="seq2seq best.ckpt to warm-start the encoder")
    p.add_argument("--precision")
    p.set_defaults(func=_cmd_drd2_train)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    args.func(args)


if __name__ == "__main__":
    main()
