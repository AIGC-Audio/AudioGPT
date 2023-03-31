import argparse
import torch


def main(checkpoint):
    state_dict = torch.load(checkpoint, map_location="cpu")
    if "optimizer" in state_dict:
        del state_dict["optimizer"]
    if "lr_scheduler" in state_dict:
        del state_dict["lr_scheduler"]
    torch.save(state_dict, checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    args = parser.parse_args()
    main(args.checkpoint)
