# 将MOT格式转换为idtracker.ai格式
# MOT格式：frame, id, x, y, w, h, _, _, _
# idtracker.ai格式：x1, y1, x2, y2, x3, y3, ..., xn, yn

import pandas as pd
from argparse import ArgumentParser
import csv


def parse_args():
    parser = ArgumentParser(description="Convert MOT format to idtracker.ai format")
    parser.add_argument(
        "-g", "--gt", help="Path to the ground truth file", required=True
    )
    parser.add_argument("-o", "--output", help="Path to the output file", required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    # gt.columns = ['frame', 'id', 'x', 'y', 'w', 'h', '_', '_', '_']

    gt = pd.read_csv(args.gt, header=None)
    gt.columns = ["frame", "id", "x", "y", "w", "h", "_", "_", "_"]

    ids = gt["id"].unique()
    start_f, end_f = gt["frame"].min(), gt["frame"].max()

    # Generate idtracker.ai dataframe
    columns = []
    for idx in ids:
        columns.extend(["x" + str(idx), "y" + str(idx)])

    idtracker = pd.DataFrame(
        columns=columns, index=range(start_f, end_f + 1), dtype=object
    )

    with open(args.gt, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            frame, obj_id, x, y, w, h, _, _, _ = row

            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)

            cx, cy = x + w / 2, y + h / 2

            idtracker.at[int(frame), "x" + str(obj_id)] = cx
            idtracker.at[int(frame), "y" + str(obj_id)] = cy

    # Export empty cells as literal "NaN"
    idtracker.to_csv(args.output, index=False, na_rep="NaN")


if __name__ == "__main__":
    main()
