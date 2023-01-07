#!/usr/bin/python3

import numpy as np
import argparse
from pathlib import Path
import cv2

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True, help="input file")
    # parser.add_argument("--out", required=True, help="output file")
    args = parser.parse_args()

    input = Path(args.infile)

    if not input.exists():
        print("input file not found")
        exit(0)

    cap = cv2.VideoCapture(str(input))

    if not cap.isOpened():
        print("Error opening video file")
        exit()

    frame_width = 1280
    frame_height = 1024

    # enc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    enc = cv2.VideoWriter_fourcc(*"XVID")
    vid_format = ".avi"
    left_out_path = input.parent / (input.with_suffix("").name + "_left" + vid_format)
    right_out_path = input.parent / (input.with_suffix("").name + "_right" + vid_format)
    print(f"writing to {left_out_path}")
    print(f"writing to {right_out_path}")

    left_out = cv2.VideoWriter(str(left_out_path), enc, 25, (frame_width, frame_height))
    right_out = cv2.VideoWriter(str(right_out_path), enc, 25, (frame_width, frame_height))

    while cap.isOpened():

        ret, frame = cap.read()

        if ret:
            left = frame[:, :1280, :]
            right = frame[:, 1280:, :]
            left_out.write(left)
            right_out.write(right)
        else:
            break

        # print(left.shape)
        # print(right.shape)

        # break
        # if ret:
        #     cv2.imshow("Frame", frame)
        #     if cv2.waitKey(25) & 0xFF == ord("q"):
        #         break
        # else:
        #     break

    cap.release()
    left_out.release()
    right_out.release()
    cv2.destroyAllWindows()

    print("finish processing videos")
