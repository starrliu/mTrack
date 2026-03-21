import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from argparse import ArgumentParser
from mtrack.post import GtFileGenerator
import json


def parse_args():
    parser = ArgumentParser(
        description="Generate corrected ID trajectory file based on mTrack outputs and dataset config. "
    )
    parser.add_argument(
        "-c", "--config", required=True, help="Path to dataset config file"
    )
    parser.add_argument(
        "-d",
        "--dump_path",
        required=True,
        help="Path to folder with mTrack output files",
    )
    parser.add_argument(
        "-g", "--gtfile", required=True, help="Output trajectory(gt) file path"
    )
    parser.add_argument(
        "-back_off",
        "--back_off",
        action="store_true",
        help="Disable backward identification",
    )
    parser.add_argument(
        "-non_id",
        "--non_identities",
        action="store_true",
        help="Retain unassigned ID trajectories",
    )
    return parser.parse_args()


if __name__ == "__main__":
    conf = parse_args()

    with open(conf.config, "r") as f:
        config = json.load(f)
        dump_path = conf.dump_path
        tmp_tags = config["tags"]
        tags = {}
        for tag, cvid in tmp_tags.items():
            tags[bytes.fromhex(tag)] = int(cvid)

    if not os.path.exists(os.path.dirname(conf.gtfile)):
        os.makedirs(os.path.dirname(conf.gtfile))

    gtfile = GtFileGenerator(tags, dump_path, conf.gtfile)

    gtfile.generate(conf.non_identities, not conf.back_off)
