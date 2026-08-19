#!/usr/bin/env python3

import os
import sys
import time
import shutil
import traceback
import importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import psutil
import numpy as np
from tqdm import tqdm

from _shared import (
    load_config,
    get_page_num,
    latest_dst_exists,
    remove_done_files,
)


# --- Setup -------------------------------------------------------------------
# os.chdir(Path(__file__).resolve().parent)
src = Path("065-remove-page-borders")
dst = Path(Path(__file__).stem)


# --- Settings ----------------------------------------------------------------
config = load_config()


# --- Helper: level adjustment (contrast stretch) -----------------------------
def apply_level(img: np.ndarray, low: float = 0.2, high: float = 0.9) -> np.ndarray:
    """
    Apply a linear color level adjustment using fractional thresholds.
    low/high are in [0,1], e.g., 0.2 = 20%, 0.9 = 90%.
    """
    if img.dtype == np.uint8:
        max_val = 255
    elif img.dtype == np.uint16:
        max_val = 65535
    else:
        raise ValueError(f"Unsupported image dtype: {img.dtype}")

    # Convert fractional thresholds to absolute values
    low_val = low * max_val
    high_val = high * max_val

    if high_val <= low_val:
        return img.copy()

    img_stretched = (img.astype(float) - low_val) * (max_val / (high_val - low_val))
    img_stretched = np.clip(img_stretched, 0, max_val)
    return img_stretched.astype(img.dtype)


# --- Worker ------------------------------------------------------------------
def process_image(image_path: Path) -> str:
    filename = image_path.name
    page_num = get_page_num(filename)
    output_path = dst / filename
    r'''
    if output_path.exists():
        output_path.unlink()
    '''

    # Load
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"error: failed to read {image_path}")
        sys.exit(1)

    # Level (contrast stretch)
    if config.do_level:

        is_image_page = False

        if isinstance(config.image_pages, list):
            if page_num in config.image_pages:
                is_image_page = True
        elif isinstance(config.image_pages, str):
            if config.image_pages == "all":
                is_image_page = True

        # TODO use the OCR result to separate text and image regions
        if is_image_page:
            lowthresh, highthresh = config.images_lowthresh, config.images_highthresh
        else:
            lowthresh, highthresh = config.text_lowthresh, config.text_highthresh

        img = apply_level(img, lowthresh, highthresh)

    # Save image
    cv2.imwrite(str(output_path), img)
    # print(f"writing {output_path}")


def try_process_image(*args):
    "ensure all exceptions are caught and serialized safely back to the main process"
    try:
        process_image(*args)
        return None
    except Exception as e:
        tb = traceback.format_exc()
        return (e, tb)


# --- Parallel execution ------------------------------------------------------
def main():

    if not config.do_level:
        print("not leveling")
        if dst.exists():
            print("keeping dst")
            return
        print("creating symlink from dst to src")
        dst.symlink_to(src)
        return

    dst.mkdir(parents=True, exist_ok=True)

    t1 = time.time()
    images = sorted(src.glob(f"*.{config.scan_format}"))
    if not images:
        print("No input files found.")
        exit(0)

    images = remove_done_files(images, dst)
    if not images:
        print("nothing to do")
        return

    # dst = OUTPUT_DIR
    files = images
    content_files = []
    extra_files = []
    for f in files:
        page_num = get_page_num(f)
        if 1 <= page_num <= config.num_pages:
            # process content pages
            content_files.append(f)
        else:
            # copy extra pages: book cover, etc
            extra_files.append(f)
    # copy extra pages: book cover, etc
    if extra_files:
        print(f"copying {len(extra_files)} extra pages")
        for f in extra_files:
            f_dst = dst / f.name
            shutil.copy(f, f_dst)
    # process only content files
    files = content_files
    images = files

    num_workers = psutil.cpu_count(logical=False) or 1
    print(f"Using {num_workers} workers...")

    tqdm_kwargs = dict(
        total=len(images),
        ncols=80,
        unit="page",
    )

    with (
        ProcessPoolExecutor(max_workers=num_workers) as executor,
        tqdm(**tqdm_kwargs) as pbar,
    ):
        futures = {executor.submit(try_process_image, img): img for img in images}
        for future in as_completed(futures):
            err = future.result()
            if err:
                executor.shutdown(cancel_futures=True)
                e, tb = err
                print(f"\nException in worker:\n{tb}")
                raise e
            pbar.update(1)

    t2 = time.time()
    print(f"done {len(images)} pages in {int(t2 - t1)} seconds using {num_workers} workers")


if __name__ == "__main__":
    main()
