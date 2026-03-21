import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(
        description="Evaluate the accuracy of the predicted trajectory."
    )
    parser.add_argument(
        "-g", "--gt", type=str, required=True, help="The path to the ground truth csv."
    )
    parser.add_argument(
        "-p", "--pred", type=str, required=True, help="The path to the predicted csv."
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="The path to the output csv."
    )
    parser.add_argument(
        "-t",
        "--thres_p",
        type=int,
        default=30,
        help="The threshold of the distance between two positions.",
    )
    parser.add_argument(
        "--direct_map", action="store_true", help="Directly map the IDs."
    )
    parser.add_argument(
        "--first_success",
        action="store_true",
        help="Evaluate the accuracy after the first success of matching.",
    )

    return parser.parse_args()


def is_correct(pos1, pos2, thres_p=30):
    """
    Check if the two positions are the same.
    p2m = 0.001 => thres_p = 30 pixels = 3 cm
    """

    # If pos1 or pos2 is NaN, return False
    if (
        pd.isnull(pos1[0])
        or pd.isnull(pos1[1])
        or pd.isnull(pos2[0])
        or pd.isnull(pos2[1])
    ):
        return False

    dis = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    return dis < thres_p


def generate_idmap(gt_df: pd.DataFrame, pred_df: pd.DataFrame, thres_p=30, frames=2000):
    """
    Generate the mapping between the ground truth and the prediction.
    """

    # Step 1: Extract the ID List
    columns = gt_df.columns.tolist()
    gtids = []
    for col in columns:
        if col.startswith("x"):
            gtids.append(int(col[1:]))

    columns = pred_df.columns.tolist()
    predids = []
    for col in columns:
        if col.startswith("x"):
            predids.append(int(col[1:]))

    # Step 2: Find the first initial predicted id.
    num_gt_ids = len(gtids)
    first_pred_ids = []
    for idx, row in pred_df.iterrows():
        for predid in predids:
            if predid in first_pred_ids:
                continue
            if not pd.isnull(row["x" + str(predid)]) and not pd.isnull(
                row["y" + str(predid)]
            ):
                # Check the number of non-NaN values
                len_first_pred_id = len(pred_df[pred_df["x" + str(predid)].notnull()])
                if len_first_pred_id < 10:
                    continue
                first_pred_ids.append(predid)
                if len(first_pred_ids) == num_gt_ids:
                    break
        if len(first_pred_ids) == num_gt_ids:
            break

    # Step 2: Generate the mapping
    idmap = {}
    tmp_gtids = gtids.copy()
    for predid in first_pred_ids:
        idmap_scores = {}
        for gtid in tmp_gtids:
            total = 0
            correct = 0
            for idx, row in gt_df.iterrows():
                if idx >= frames:
                    break
                gt_pos = [row["x" + str(gtid)], row["y" + str(gtid)]]
                pred_pos = [
                    pred_df["x" + str(predid)][idx],
                    pred_df["y" + str(predid)][idx],
                ]

                if (
                    pd.isnull(gt_pos[0])
                    or pd.isnull(gt_pos[1])
                    or pd.isnull(pred_pos[0])
                    or pd.isnull(pred_pos[1])
                ):
                    continue

                total += 1
                if is_correct(gt_pos, pred_pos, thres_p):
                    correct += 1

            if total == 0:
                idmap_scores[gtid] = 0
            else:
                idmap_scores[gtid] = correct / total

        # Step 3: Find the best mapping
        best_gtid = max(idmap_scores, key=idmap_scores.get)
        idmap[best_gtid] = predid
        tmp_gtids.remove(best_gtid)
        print(
            f"GT ID {best_gtid} is mapped to Pred ID {predid} with a score of {idmap_scores[best_gtid]}."
        )

    return idmap


def evaluate_id_acc(
    gt_df: pd.DataFrame, pred_df: pd.DataFrame, idmap: dict, thres_p=30
):
    """
    Evaluate the accuracy of the predicted trajectory.
    """

    # Step 1: Extract the ID List
    columns = gt_df.columns.tolist()
    gtids = []
    for col in columns:
        if col.startswith("x"):
            gtids.append(int(col[1:]))

    # Step 2: Iterate through the gt_df and pred_df
    wrong_identified = 0
    non_identified = 0
    total_num = 0

    for idx, row in gt_df.iterrows():
        for gtid in gtids:
            gt_pos = [row["x" + str(gtid)], row["y" + str(gtid)]]
            predid = idmap[gtid]

            if pd.isnull(gt_pos[0]) or pd.isnull(gt_pos[1]):
                continue

            total_num += 1

            if (
                "x" + str(predid) not in pred_df.columns
                or "y" + str(predid) not in pred_df.columns
            ):
                non_identified += 1
                continue

            if len(pred_df) == 0 or idx > pred_df.index[-1]:
                non_identified += 1
                continue

            pred_pos = [
                pred_df["x" + str(predid)][idx],
                pred_df["y" + str(predid)][idx],
            ]

            if pd.isnull(pred_pos[0]) or pd.isnull(pred_pos[1]):
                non_identified += 1
                # print(f"Frame {idx+1}: GT ID {gtid} is not identified.")
                continue

            if not is_correct(gt_pos, pred_pos, thres_p):
                wrong_identified += 1
                continue

    mis_rate = wrong_identified / total_num
    non_rate = non_identified / total_num
    acc = 1 - mis_rate - non_rate

    return mis_rate, non_rate, acc


def evaluate_id_acc_from_first_success(
    gt_df: pd.DataFrame, pred_df: pd.DataFrame, idmap: dict, thres_p=30
):
    """
    Evaluate the accuracy of the predicted trajectory from the first success.

    Algorithm:
    1. Find the first frame in pred_df that all the ids from gt_df are identified.
    2. Evaluate the accuracy of the predicted trajectory from the first success.

    Args:
        gt_df (pd.DataFrame): The ground truth dataframe.
        pred_df (pd.DataFrame): The predicted dataframe.
        idmap (dict): The mapping between the ground truth and the prediction.
        thres_p (int, optional): The threshold of the distance between two positions. Defaults to 30.
    """

    # Step 1: Extract the ID List
    columns = gt_df.columns.tolist()
    gtids = []
    for col in columns:
        if col.startswith("x"):
            gtids.append(int(col[1:]))

    # Step 2: Find the first frame where all IDs are successfully identified
    first_success_frame = None
    for idx, row in pred_df.iterrows():
        all_identified = True
        for gtid in gtids:
            predid = idmap[gtid]

            # Check if the predicted ID exists in the dataframe
            if (
                "x" + str(predid) not in pred_df.columns
                or "y" + str(predid) not in pred_df.columns
            ):
                all_identified = False
                break

            # Check if the position is not NaN
            pred_pos = [row["x" + str(predid)], row["y" + str(predid)]]
            if pd.isnull(pred_pos[0]) or pd.isnull(pred_pos[1]):
                all_identified = False
                break

        if all_identified:
            first_success_frame = idx
            break

    if first_success_frame is None:
        print("Warning: No frame found where all IDs are successfully identified.")
        return 0.0, 1.0, 0.0  # mis_rate, non_rate, acc

    print(f"First success frame: {first_success_frame}")

    # Step 3: Evaluate accuracy from the first success frame onwards
    gt_df_from_success = gt_df.iloc[first_success_frame:]
    pred_df_from_success = pred_df.iloc[first_success_frame:]

    return evaluate_id_acc(gt_df_from_success, pred_df_from_success, idmap, thres_p)


if __name__ == "__main__":
    args = parse_args()

    # Step 1: Load the csv files
    gt_df = pd.read_csv(args.gt)
    pred_df = pd.read_csv(args.pred)

    # Step 2: Generate ID mapping
    if not args.direct_map:
        idmap = generate_idmap(gt_df, pred_df, args.thres_p)
    else:
        idmap = {}
        columns = gt_df.columns.tolist()
        gtids = []
        for col in columns:
            if col.startswith("x"):
                gtids.append(int(col[1:]))

        for gtid in gtids:
            idmap[gtid] = gtid

    # Step 3: Evaluate the accuracy based on the mode
    if args.first_success:
        # Evaluate from first success
        mis_rate, non_rate, acc = evaluate_id_acc_from_first_success(
            gt_df, pred_df, idmap, args.thres_p
        )
    else:
        # Evaluate overall accuracy
        mis_rate, non_rate, acc = evaluate_id_acc(gt_df, pred_df, idmap, args.thres_p)

    # Step 4: Save the result
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result = {
        "mis_rate": mis_rate,
        "non_rate": non_rate,
        "acc": acc,
        "thres_p": args.thres_p,
    }

    with open(args.output, "w") as f:
        json.dump(result, f)
