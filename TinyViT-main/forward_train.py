import argparse
import torch
import torch.cuda.profiler as ncu
from config import get_config
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--accumulation-steps", type=int, default=None)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--only-cpu", action="store_true")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    model = model.to(device).train()

    dummy_input = torch.randn(
        config.DATA.BATCH_SIZE, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE
    ).to(device)

    torch.cuda.synchronize()
    ncu.start()
    with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
        _ = model(dummy_input)
    ncu.stop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
