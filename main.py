import argparse

from experiment import Experiment


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="YAML config file")
    parser.add_argument("-s", "--src", type=str, default=None)
    parser.add_argument("-g", "--gpu", action="store_true", default=False)
    parser.add_argument("-t", "--train", action="store_true", default=True)
    parser.add_argument("-i", "--inference", action="store_true", default=False)
    parser.add_argument("-e", "--eval", action="store_true", default=False)
    parser.add_argument("-r", "--resume", action="store_true", default=False)
    parser.add_argument(
        "--wandb", action="store_true", help="Log run to Weights and Biases."
    )

    return parser


def main(args):
    experiment = Experiment(args)
    if args.train:
        experiment.train()

    if args.eval:
        experiment.eval()

    if args.inference:
        experiment.inference()


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()

    main(args)
