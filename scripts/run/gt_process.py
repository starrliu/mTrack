import argparse
import subprocess


def remove_dup_id(input_file, output_file):
    return subprocess.run(
        [
            "python",
            "./scripts/run/remove_dup_id.py",
            "-i",
            input_file,
            "-o",
            output_file,
        ],
        check=True,
    )


def interpolate(input_file, output_file, stack_size):
    return subprocess.run(
        [
            "python",
            "./scripts/run/interpolate_gt.py",
            "-gt",
            input_file,
            "-o",
            output_file,
            "-s",
            stack_size,
        ],
        check=True,
    )


def convert_format(input_file, output_file):
    return subprocess.run(
        [
            "python",
            "./scripts/run/convert_format.py",
            "-g",
            input_file,
            "-o",
            output_file,
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run ground-truth preprocessing pipeline."
    )
    parser.add_argument("-i", "--input", required=True, help="Input file path")
    parser.add_argument(
        "-o1",
        "--output1",
        required=True,
        help="Output path for deduplicated and interpolated MOT-format data",
    )
    parser.add_argument(
        "-o2",
        "--output2",
        required=True,
        help="Output path for converted idtracker-format data",
    )
    parser.add_argument(
        "-s",
        "--stack_size",
        required=False,
        type=int,
        default=5,
        help="Stack size value for interpolation",
    )

    args = parser.parse_args()

    ret = remove_dup_id(args.input, args.output1)
    if ret.returncode != 0:
        print("remove_dup_id failed")
        return
    ret = interpolate(args.output1, args.output1, str(args.stack_size))
    if ret.returncode != 0:
        print("interpolate failed")
        return
    ret = convert_format(args.output1, args.output2)
    if ret.returncode != 0:
        print("convert_format failed")
        return


if __name__ == "__main__":
    main()
