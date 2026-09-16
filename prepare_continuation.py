"""Fork a scale checkpoint into a separate full-state continuation run."""
import argparse
import shlex

from para_audio_id.audio_lm.continuation import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--additional-steps", type=int, default=25000)
    parser.add_argument("--target-lr", type=float, default=5e-5)
    parser.add_argument("--ramp-steps", type=int, default=2000)
    args = parser.parse_args()
    path, cfg = prepare(
        args.checkpoint, args.run_id, additional_steps=args.additional_steps,
        target_lr=args.target_lr, ramp_steps=args.ramp_steps,
    )
    print(f"Prepared {path}; endpoint {cfg['train']['max_steps']}")
    print(shlex.join([
        "python", "train.py", str(path.parent / "continuation.yaml"),
        "--resume", "--ckpt-path", str(path), "--run-id", args.run_id,
        "--wandb-online",
    ]))


if __name__ == "__main__":
    main()
