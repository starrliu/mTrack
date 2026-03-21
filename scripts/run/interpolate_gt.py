import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from mtrack.post import InterpolationGtFile
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-gt", "--gt", dest="gt", required=True)
    parser.add_argument("-o", "--output", dest="output", required=True)
    parser.add_argument(
        "-s", "--stack_size", dest="stack_size", required=False, default=5, type=int
    )

    return parser.parse_args()


def main():
    args = parse_args()

    gt_txt = args.gt
    output_file = args.output
    stack_size = args.stack_size

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    igf = InterpolationGtFile(gt_txt, output_file, stack_size)

    igf.interpolate()


if __name__ == "__main__":
    main()
