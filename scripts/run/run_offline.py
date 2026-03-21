"""Script for offline inference using mTrack."""

# pylint: disable=no-member
import os, sys
import traceback
import warnings
from argparse import ArgumentParser

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2
import pandas as pd

from mtrack.config import load_config as load_mtrack_config
from mtrack.data.load import load_config as load_dataset_config
from mtrack.data.load import load_object_trace, load_tag_data
from mtrack.dump import MTrackDumper
from mtrack.mtrack import MTrack
from mtrack.simulate.image_loader import ImageLoader
from mtrack.simulate.simulate import SimulatorFromData

warnings.simplefilter(action="ignore", category=FutureWarning)


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-rf",
        "--rfid-data",
        dest="rfid_data",
        required=True,
        help="Path to the RFID data CSV file",
    )
    parser.add_argument(
        "-cam",
        "--camera-data",
        dest="camera_data",
        required=True,
        help="Path to the camera data CSV file",
    )
    parser.add_argument(
        "-g", "--gt", dest="gt", required=True, help="Path to the trajectory file"
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        required=True,
        help="Path to the dataset config file",
    )
    parser.add_argument(
        "--mtrack-config",
        dest="mtrack_config",
        default=None,
        help="Path to the MTrack config file (YAML)",
    )
    parser.add_argument(
        "-i",
        "--img",
        dest="img",
        default=None,
        help="Directory containing the images (required when --verbose is enabled)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        required=True,
        help="Directory to save output files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable visualization",
    )
    parser.add_argument(
        "--no-checker",
        dest="no_checker",
        action="store_true",
        help="Disable global checker",
    )
    parsed_args = parser.parse_args()

    if parsed_args.verbose and not parsed_args.img:
        parser.error("--img is required when --verbose is enabled")

    return parsed_args


if __name__ == "__main__":
    args = parse_args()

    rfcsv = args.rfid_data
    camera_csv = args.camera_data
    gt_csv = args.gt

    dataset_config = load_dataset_config(args.config)
    antpos = dataset_config["ant_pos"]
    tags = dataset_config["tags"]
    p2m = dataset_config["p2m"]

    mtrack_config = None
    if args.mtrack_config:
        mtrack_config = load_mtrack_config(args.mtrack_config)
        print(f"[INFO] Using MTrack config: {args.mtrack_config}")

    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dumper = MTrackDumper(output_dir, video=args.verbose)

    print("[INFO] Loading data...")
    rfdata = load_tag_data(rfcsv)
    objdata = load_object_trace(gt_csv, camera_csv)
    cameradf = pd.read_csv(camera_csv)

    simulator = SimulatorFromData(rfdata, objdata, cameradf)

    image_loader = None
    if args.verbose:
        image_loader = ImageLoader(args.img)

    print("[INFO] Initializing MTrack...")
    start_t = pd.Timestamp(cameradf["TimeStamp"].min())
    end_t = pd.Timestamp(cameradf["TimeStamp"].max())

    simulator.set_time_range(start_t, end_t)

    mtrack = MTrack(
        tags=tags, antpos=antpos, p2m=p2m, config=mtrack_config, no_sel=True
    )

    frame_cnt = 0

    print("[INFO] Starting tracking...")
    is_read = True
    while is_read:
        try:
            res = simulator.read_data()
            if res is None:
                print("[INFO] End of data reached")
                is_read = False
            else:
                frame_cnt += 1
                rfid_messages, tracker_message = res

                mtrack_res = mtrack.track(
                    tracker_message, rfid_messages, global_check=not args.no_checker
                )
                dumper.dump_mtrack_res(
                    mtrack_res.frame,
                    tracker_message.timestamp,
                    mtrack_res.track_res,
                    mtrack_res.matched_ids,
                    mtrack_res.bc_res,
                    mtrack_res.mismatched_tags,
                )
                dumper.dump_mtrack_state(frame_cnt, mtrack.snapshot())
                if mtrack.select is not None:
                    dumper.dump_mtrack_select_trace(
                        frame_cnt,
                        mtrack.select.current_reading_state,
                        list(mtrack.select.reading_tags.keys()),
                    )

                if args.verbose:
                    img = image_loader.get_image(frame_cnt)
                    img = mtrack.annotate_frame(img)
                    cv2.putText(
                        img,
                        f"Frame: {frame_cnt}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        img,
                        f"Time: {tracker_message.timestamp}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )

                    dumper.dump_img(img)

                    cv2.imshow("MTrack", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        is_read = False
                        break

                if frame_cnt % 1000 == 0:
                    print(f"[INFO] Processed {frame_cnt} frames")

        except Exception as e:
            print(f"[ERROR] Exception occurred: {e}")
            traceback.print_exc()
            break

    dumper.close()
    mtrack.close()
    print(f"[INFO] Tracking completed. Total frames processed: {frame_cnt}")
