import pandas as pd
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input", dest="input", help="input file", type=str, required=True
    )
    parser.add_argument(
        "-o", "--output", dest="output", help="output file", type=str, required=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 读取原始数据文件
    input_file = args.input
    output_file = args.output

    # 读取数据
    df = pd.read_csv(input_file, header=None)

    # 按帧分组并移除同一帧内重复的ID
    df_unique = df.drop_duplicates(subset=[0, 1])

    # 保存到新的文件
    df_unique.to_csv(output_file, header=False, index=False)
