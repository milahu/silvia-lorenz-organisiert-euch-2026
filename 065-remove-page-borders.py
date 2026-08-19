#!/usr/bin/env python3
"""
extract scanned page from gray background

restore the binding edge of the page
by filling the missing width
with the average color near the binding edge
"""

from pathlib import Path

INPUT_DIR = Path("060-rotate-crop")
OUTPUT_DIR = Path("065-remove-page-borders")

# === Tuning parameters ===
DEBUG = True
DEBUG = False
BORDER_SIZE = 100  # pixels

RANSAC_ITER = 400
RANSAC_INLIER_DIST = 6.0
# RANSAC_MIN_INLIERS = 30
RANSAC_MIN_INLIERS = 20

THRESH_HIGH_PERCENTILE = 99
THRESH_MIN = 200

# cleanup the inside edge
#
# the "dirty" inside edge is created
# by rotating and warping the three other edges
# so in many cases, the inside edge is crooked (not vertical)
#
# if we remove the dirty inside edge
# then we remove a small rectangle from the inside edge
# to get a straight inside edge (vertical)
remove_inside_transparent_strip = True
#
# FIXME this produces black triangles on the inside edges
# if we keep the dirty inside edge
# then there is a small transparent rectangle inside of the inside edge
# remove_inside_transparent_strip = False



# 8c. remove small gray artifacts along page edges

WHITE_BORDER_WIDTH = 10
WHITE_TEST_INNER = 10
WHITE_TEST_OUTER = 20
WHITE_FRACTION_THRESHOLD = 0.99
PRACTICALLY_WHITE_THRESHOLD = 250  # for uint8 images



import os
import re
import math
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2
import PIL.Image
import psutil
from tqdm import tqdm


from _shared import (
    load_config,
    get_page_num,
    latest_dst_exists,
    remove_done_files,
)

config = load_config()

# no, this is wrong if (config.do_rotate == True)
# scan_x = config.scan_x
# scan_y = config.scan_y
# config.scan_aspect = scan_x / scan_y
# ASPECT = config.scan_aspect

ASPECT = config.rotated_scan_aspect


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_dbg(img, path):
    ensure_dir(path.parent)
    save_image(path, img)

def percentile_threshold(gray):
    high_p = np.percentile(gray, THRESH_HIGH_PERCENTILE)
    thr = max(THRESH_MIN, int(high_p * 0.95))
    _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    return mask, thr, int(high_p)

def detect_vertical_streaks(mask, approx_width=3, length_thresh_ratio=0.15):
    h, w = mask.shape
    kx = approx_width
    ky = max(15, int(h * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    long_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(long_vertical, connectivity=8)
    streak_mask = np.zeros_like(mask)
    length_thresh = max(10, int(h * length_thresh_ratio))
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if hh >= length_thresh and ww <= max(5, int(w * 0.01)):
            streak_mask[labels == i] = 255
    return streak_mask

def keep_largest_component(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out

def contour_to_pts(contour):
    return contour.reshape(-1,2)

def fit_line_ransac(pts, iterations=RANSAC_ITER, inlier_dist=RANSAC_INLIER_DIST, min_inliers=RANSAC_MIN_INLIERS):
    if len(pts) < 2:
        raise ValueError("Not enough points")
    best_inliers = None
    best_model = None
    n = len(pts)
    ptsf = pts.astype(np.float32)
    for _ in range(iterations):
        i1, i2 = random.sample(range(n), 2)
        p1 = ptsf[i1]; p2 = ptsf[i2]
        vx = float(p2[0] - p1[0]); vy = float(p2[1] - p1[1])
        if vx == 0 and vy == 0:
            continue
        dists = np.abs(vy*(ptsf[:,0]-p1[0]) - vx*(ptsf[:,1]-p1[1])) / (math.hypot(vx, vy) + 1e-12)
        inliers = dists <= inlier_dist
        cnt = int(inliers.sum())
        if cnt >= min_inliers and (best_inliers is None or cnt > int(best_inliers.sum())):
            best_inliers = inliers.copy()
            best_model = (vx, vy, float(p1[0]), float(p1[1]))
    if best_model is None:
        vx, vy, x0, y0 = cv2.fitLine(ptsf, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        inlier_mask = np.ones(len(pts), dtype=bool)
        return float(vx), float(vy), float(x0), float(y0), inlier_mask
    inlier_pts = ptsf[best_inliers]
    vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    inlier_mask = best_inliers
    return float(vx), float(vy), float(x0), float(y0), inlier_mask

def intersect_lines(l1, l2):
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
    b = np.array([x2 - x1, y2 - y1], dtype=np.float32)
    det = np.linalg.det(A)
    if abs(det) < 1e-8:
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    t1, t2 = np.linalg.solve(A, b)
    xi = x1 + t1 * vx1
    yi = y1 + t1 * vy1
    return float(xi), float(yi)

def intersect_line_with_vertical_boundary(line, x):
    """
    Intersect a parametric line

        (x0, y0) + t * (vx, vy)

    with the vertical line

        x = constant

    Returns (x, y).
    """
    vx, vy, x0, y0 = map(float, line)
    if abs(vx) < 1e-12:
        raise ValueError("Line is parallel to vertical boundary")
    t = (x - x0) / vx
    y = y0 + t * vy
    return np.array([x, y], dtype=np.float32)


def build_affine_without_shear(pt_v_top, pt_v_bot, expected_w, expected_h, bad_on_left, img, dbgdir=None):
    """
    Build a source triangle that enforces orthogonal page axes (no shear) using:
      - pt_v_top: intersection of top line and good vertical (top-right when good vertical is right)
      - pt_v_bot: intersection of bottom line and good vertical (bottom-right when good vertical is right)
    Returns (M_aff, src_corners_dict, dst_corners_dict, reason)
    - M_aff: 2x3 affine matrix or None on failure
    - src_corners_dict: dict of TL, TR, BL, BR (np.float32 2-vectors)
    - dst_corners_dict: dict of TL, TR, BL, BR in destination coordinates
    - reason: diagnostic string
    """
    # convert to vectors
    pt_v_top = np.array(pt_v_top, dtype=np.float32)
    pt_v_bot = np.array(pt_v_bot, dtype=np.float32)

    # compute direction along top edge: vector from bottom intersection to top intersection projected horizontally
    # but we don't have explicit top_line direction vector here; we will approximate using (pt_v_top - pt_v_bot) rotated?
    # Better: caller should supply top_line direction (vx,vy). If you don't have it, approximate from nearby contour points.
    # For compatibility with your code, expect you have top_vx, top_vy; if not, fall back to vector between two top contour points.
    # Here we'll assume top_vx, top_vy are available in outer scope; otherwise compute from pt_v_top->pt_v_bot displacement projected perpendicular:
    # To keep this helper self-contained, we compute top_dir as average of (pt_v_top_to_some_right) - but simplest robust approach:
    # compute approximate top direction by taking small offset vector along top by sampling image gradient: fallback to (1,0).

    # --- Here caller should provide top_unit; we'll compute a safe top_unit using neighbor pixels if available ---
    # We'll attempt to derive a top_unit by sampling a short step along the image: use vector from pt_v_top to pt_v_top projected to image center
    cx, cy = img.shape[1] / 2.0, img.shape[0] / 2.0
    # prefer vector pointing rightwards by projecting displacement to center
    approx = pt_v_top - np.array([cx, cy], dtype=np.float32)
    if np.linalg.norm(approx) < 1e-6:
        approx = np.array([1.0, 0.0], dtype=np.float32)
    ux, uy = approx / (np.linalg.norm(approx) + 1e-12)

    # Force ux positive (pointing to the right) — we want u to be rightward
    if ux < 0:
        ux, uy = -ux, -uy

    # make perpendicular downwards: p = (-uy, ux)
    px, py = -uy, ux

    # normalize again (just in case)
    n_u = math.hypot(ux, uy)
    n_p = math.hypot(px, py)
    if n_u < 1e-6 or n_p < 1e-6:
        return None, None, None, "degenerate_axes"
    ux, uy = ux / n_u, uy / n_u
    px, py = px / n_p, py / n_p

    # Now construct corners. We know pt_v_top/pt_v_bot correspond to the known vertical edge:
    # If bad_on_left is True, good vertical is the RIGHT edge => pt_v_top == TR, pt_v_bot == BR.
    # If bad_on_left is False, good vertical is LEFT edge => pt_v_top == TL, pt_v_bot == BL.
    if bad_on_left:
        src_TR = pt_v_top
        src_BR = pt_v_bot
        # compute TL and BL by moving left along top unit by expected_w
        src_TL = src_TR - np.array([ux, uy], dtype=np.float32) * float(expected_w)
        src_BL = src_BR - np.array([ux, uy], dtype=np.float32) * float(expected_w)
    else:
        src_TL = pt_v_top
        src_BL = pt_v_bot
        # compute TR and BR by moving right along top unit by expected_w
        src_TR = src_TL + np.array([ux, uy], dtype=np.float32) * float(expected_w)
        src_BR = src_BL + np.array([ux, uy], dtype=np.float32) * float(expected_w)

    # Optional: enforce vertical height by projecting (TL->BL) onto perp and rescale to expected_h
    # Compute current vertical vector and its projection onto perp (px,py)
    cur_v = src_BL - src_TL
    proj = cur_v.dot(np.array([px, py], dtype=np.float32))
    # if projection is tiny, fall back but otherwise adjust BL to ensure exact page height
    if abs(proj) > 1e-6:
        # adjust scale to exact expected_h
        scale = float(expected_h) / proj
        # recompute BL, BR as TL + perp*expected_h and TR + perp*expected_h
        src_BL = src_TL + np.array([px, py], dtype=np.float32) * float(expected_h)
        src_BR = src_TR + np.array([px, py], dtype=np.float32) * float(expected_h)
    else:
        # if proj nearly zero, we cannot rely on vertical measurement; still use perpendicular step
        src_BL = src_TL + np.array([px, py], dtype=np.float32) * float(expected_h)
        src_BR = src_TR + np.array([px, py], dtype=np.float32) * float(expected_h)

    # Build src/dst dicts
    src_corners = {
        "TL": src_TL.astype(np.float32),
        "TR": src_TR.astype(np.float32),
        "BL": src_BL.astype(np.float32),
        "BR": src_BR.astype(np.float32)
    }
    dst_corners = {
        "TL": np.array([0.0, 0.0], dtype=np.float32),
        "TR": np.array([expected_w - 1.0, 0.0], dtype=np.float32),
        "BL": np.array([0.0, expected_h - 1.0], dtype=np.float32),
        "BR": np.array([expected_w - 1.0, expected_h - 1.0], dtype=np.float32)
    }

    # choose triangle for affine depending on orientation (map TL, BL, TR -> dst TL, BL, TR)
    src_tri = np.vstack([src_corners["TL"], src_corners["BL"], src_corners["TR"]]).astype(np.float32)
    dst_tri = np.vstack([dst_corners["TL"], dst_corners["BL"], dst_corners["TR"]]).astype(np.float32)

    # validate triangle
    area = abs(0.5 * (src_tri[0,0]*(src_tri[1,1]-src_tri[2,1]) + src_tri[1,0]*(src_tri[2,1]-src_tri[0,1]) + src_tri[2,0]*(src_tri[0,1]-src_tri[1,1])))
    if area < 1.0 or not np.isfinite(src_tri).all():
        return None, src_corners, dst_corners, f"invalid_src_tri_area={area:.3f}"

    try:
        M = cv2.getAffineTransform(src_tri, dst_tri)
    except cv2.error as e:
        return None, src_corners, dst_corners, f"getAffineTransform_failed:{e}"

    # debug overlay: draw corners & axes if dbgdir provided
    if dbgdir is not None:
        vis = img.copy()
        def draw_pt(pt, color=(0,0,255), tag=""):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, color, -1)
            if tag:
                cv2.putText(vis, tag, (int(pt[0]+6), int(pt[1]-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        draw_pt(src_corners["TL"], (0,255,0), "TL")
        draw_pt(src_corners["TR"], (0,128,255), "TR")
        draw_pt(src_corners["BL"], (255,0,0), "BL")
        draw_pt(src_corners["BR"], (255,255,0), "BR")
        # draw axis arrows from TL
        origin = src_corners["TL"]
        arrow_u = origin + np.array([ux, uy], dtype=np.float32) * 80.0
        arrow_p = origin + np.array([px, py], dtype=np.float32) * 80.0
        cv2.arrowedLine(vis, (int(origin[0]), int(origin[1])), (int(arrow_u[0]), int(arrow_u[1])), (255,0,255), 2)
        cv2.arrowedLine(vis, (int(origin[0]), int(origin[1])), (int(arrow_p[0]), int(arrow_p[1])), (0,255,255), 2)
        save_image(dbgdir.joinpath("debug_axes_and_corners.png"), vis)

    return M, src_corners, dst_corners, "ok"


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def split_edge_candidates(contour, bad_on_left):
    pts = contour.reshape(-1, 2)

    xs = pts[:,0]
    ys = pts[:,1]

    w = xs.max() - xs.min()
    h = ys.max() - ys.min()

    margin_x = w * 0.15
    margin_y = h * 0.15

    # outside edge
    if bad_on_left:
        outside = pts[xs > xs.max() - margin_x]
    else:
        outside = pts[xs < xs.min() + margin_x]

    # top edge
    top = pts[ys < ys.min() + margin_y]

    # bottom edge
    bottom = pts[ys > ys.max() - margin_y]

    return top, bottom, outside


def line_angle(line):
    vx, vy, _, _ = line
    return math.atan2(vy, vx)


def horizontal_line_angle(line):
    vx, vy, _, _ = line

    a = math.degrees(math.atan2(vy, vx))

    while a < -45:
        a += 180

    while a > 45:
        a -= 180

    return a


def vertical_line_angle(line):
    vx, vy, _, _ = line

    a = math.degrees(math.atan2(vy, vx))

    while a < 45:
        a += 180

    while a > 135:
        a -= 180

    return a


def normalize_angle_deg(a):
    while a < -90:
        a += 180
    while a > 90:
        a -= 180
    return a


def get_gray_mask_contours(img, dbgdir):
    # --- threshold for geometry only ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_init, thr, hp = percentile_threshold(gray)
    if cv2.mean(gray)[0] < 127:
        mask_init = cv2.bitwise_not(mask_init)
    if DEBUG:
        save_dbg(gray, dbgdir.joinpath("01_gray.png"))
        save_dbg(mask_init, dbgdir.joinpath(f"02_thresh_thr{thr}_hp{hp}.png"))

    # without a9dc2b24e6b0d1d49b6fc232223d6431ba3442a5 bad: fix perspective transform for broken ADF scanners
    # Invert if necessary, so the page is always "white" for thresholding
    # We'll use Otsu's method to detect high-contrast area
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Determine whether the page is darker or lighter than background
    mean_val = cv2.mean(gray, mask=None)[0]
    if mean_val < 127:  # dark page: invert mask
        mask = cv2.bitwise_not(mask)

    # Morphology to remove small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return gray, mask, contours


def repair_binding(img, bad_on_left, width=50):

    h,w = img.shape[:2]

    if bad_on_left:
        sample_range = range(width)
        fill_range = range(width)
        sample_x = width
    else:
        sample_range = range(w-width,w)
        fill_range = range(w-width,w)
        sample_x = w-width

    for y in range(h):

        if bad_on_left:
            sample = img[y, width:width+20]
            color = np.mean(sample,axis=0)
            img[y,:width] = color

        else:
            sample = img[y,w-width-20:w-width]
            color = np.mean(sample,axis=0)
            img[y,w-width:] = color

    return img


def px_of_mm(mm, dpi):
    return mm * dpi / 25.4


EDGE_TOP = 0
EDGE_BOTTOM = 1
EDGE_LEFT = 2
EDGE_RIGHT = 3


def detect_edge_points(gray, mask, edge, search_size):
    h, w = mask.shape
    pts = []

    if edge == EDGE_TOP:
        y_max = min(search_size, h)
        for x in range(w):
            # TODO why 255
            ys = np.where(mask[:y_max, x] == 255)[0]
            if len(ys):
                pts.append((x, ys[0]))

    elif edge == EDGE_BOTTOM:
        y_min = max(0, h - search_size)
        for x in range(w):
            ys = np.where(mask[y_min:, x] == 255)[0]
            if len(ys):
                pts.append((x, y_min + ys[-1]))

    elif edge == EDGE_LEFT:
        x_max = min(search_size, w)
        for y in range(h):
            xs = np.where(mask[y, :x_max] == 255)[0]
            if len(xs):
                pts.append((xs[0], y))

    # elif edge == EDGE_RIGHT:
    else:
        x_min = max(0, w - search_size)
        for y in range(h):
            xs = np.where(mask[y, x_min:] == 255)[0]
            if len(xs):
                pts.append((x_min + xs[-1], y))

    pts = np.asarray(pts, np.float32)

    # Transition verification
    #
    # The first white pixel isn't always the page.
    # It might be dust, glare, or text sticking out.
    #
    # Keep only points that actually separate gray background
    # from the white page.
    pts = verify_transition(
        gray,
        pts,
        edge,
    )

    return pts


def verify_transition(gray, points, edge):
    if len(points) == 0:
        return np.empty((0,2), dtype=np.float32)

    # TODO actually refactor verify_horizontal and verify_vertical

    if edge == EDGE_TOP:
        return verify_horizontal(gray, points, top_edge=True)

    elif edge == EDGE_BOTTOM:
        return verify_horizontal(gray, points, top_edge=False)

    elif edge == EDGE_LEFT:
        return verify_vertical(gray, points, right_edge=False)

    # elif edge == EDGE_RIGHT:
    else:
        return verify_vertical(gray, points, right_edge=True)


def verify_horizontal(gray, points, top_edge):
    good = []

    H, W = gray.shape

    for x, y in points.astype(int):

        if y < 3 or y >= H-3:
            continue

        if top_edge:
            outside = np.mean(gray[y-3:y, x])
            inside  = np.mean(gray[y:y+3, x])
        else:
            outside = np.mean(gray[y:y+3, x])
            inside  = np.mean(gray[y-3:y, x])

        r'''
        # must have strong contrast
        if abs(float(inside) - float(outside)) < 40:
            continue
        '''
        r'''
        # must have strong contrast
        if top_edge:
            if not is_horizontal_transition(gray, x, y):
                continue
        else:
            if not is_vertical_transition(gray, x, y):
                continue
        '''
        # background is always gray
        # Require the outside to actually look like scanner background
        BACKGROUND_MIN = 80
        BACKGROUND_MAX = 180
        if not (BACKGROUND_MIN <= outside <= BACKGROUND_MAX):
            continue

        # outside should be scanner gray
        if not (60 < outside < 200):
            continue

        # inside should be brighter
        if inside <= outside:
            continue

        good.append((x, y))

    return np.asarray(good, np.float32)


def verify_vertical(gray, points, right_edge):
    good = []

    H, W = gray.shape

    for x, y in points.astype(int):

        if x < 3 or x >= W-3:
            continue

        if right_edge:
            outside = np.mean(gray[y, x:x+3])
            inside  = np.mean(gray[y, x-3:x])
        else:
            outside = np.mean(gray[y, x-3:x])
            inside  = np.mean(gray[y, x:x+3])

        if abs(float(inside) - float(outside)) < 40:
            continue

        if not (60 < outside < 200):
            continue

        if inside <= outside:
            continue

        good.append((x, y))

    return np.asarray(good, np.float32)


# TODO refactor x/y

def reject_outliers_horizontal(pts, tolerance=40):
    if len(pts) == 0:
        return pts

    ys = pts[:,1]
    median = np.median(ys)

    return pts[np.abs(ys - median) < tolerance]


def reject_outliers_vertical(pts, tolerance=40):
    if len(pts) == 0:
        return pts

    xs = pts[:,0]
    median = np.median(xs)

    return pts[np.abs(xs - median) < tolerance]


# TODO dedent
# these were part of "def process_image"
if 1:
    # ---------- small helpers ----------
    def ensure_dir(p):
        os.makedirs(p, exist_ok=True)

    def save_dbg(img, path):
        ensure_dir(path.parent)
        save_image(path, img)

    def percentile_threshold(gray):
        high_p = np.percentile(gray, THRESH_HIGH_PERCENTILE)
        thr = max(THRESH_MIN, int(high_p * 0.95))
        _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        return mask, thr, int(high_p)

    def keep_largest_component(mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        out = np.zeros_like(mask)
        out[labels == best] = 255
        return out

    def detect_vertical_streaks(mask, approx_width=3, length_thresh_ratio=0.15):
        h, w = mask.shape
        kx = approx_width
        ky = max(15, int(h * 0.02))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        long_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(long_vertical, connectivity=8)
        streak_mask = np.zeros_like(mask)
        length_thresh = max(10, int(h * length_thresh_ratio))
        for i in range(1, num_labels):
            x, y, ww, hh, area = stats[i]
            if hh >= length_thresh and ww <= max(5, int(w * 0.01)):
                streak_mask[labels == i] = 255
        return streak_mask

    def contour_to_pts(c):
        return c.reshape(-1, 2)

    def fit_line_ransac(pts, iterations=RANSAC_ITER, inlier_dist=RANSAC_INLIER_DIST, min_inliers=RANSAC_MIN_INLIERS):
        if len(pts) < 2:
            raise ValueError("Not enough points for line fit")
        best_inliers = None
        best_cnt = 0
        best_model = None
        ptsf = pts.astype(np.float32)
        n = len(ptsf)
        for _ in range(iterations):
            i1, i2 = random.sample(range(n), 2)
            p1 = ptsf[i1]; p2 = ptsf[i2]
            vx = float(p2[0] - p1[0]); vy = float(p2[1] - p1[1])
            if abs(vx) < 1e-6 and abs(vy) < 1e-6:
                continue
            dists = np.abs(vy*(ptsf[:,0]-p1[0]) - vx*(ptsf[:,1]-p1[1])) / (math.hypot(vx, vy) + 1e-12)
            inliers = dists <= inlier_dist
            cnt = int(inliers.sum())
            if cnt >= min_inliers and cnt > best_cnt:
                best_cnt = cnt
                best_inliers = inliers.copy()
                best_model = (vx, vy, float(p1[0]), float(p1[1]))
        if best_model is None:
            vx, vy, x0, y0 = cv2.fitLine(ptsf, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            inlier_mask = np.ones(len(ptsf), dtype=bool)
            return float(vx), float(vy), float(x0), float(y0), inlier_mask
        inlier_pts = ptsf[best_inliers]
        vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return float(vx), float(vy), float(x0), float(y0), best_inliers

    def intersect_lines(l1, l2):
        vx1, vy1, x1, y1 = l1
        vx2, vy2, x2, y2 = l2
        A = np.array([[vx1, -vx2], [vy1, -vy2]], dtype=np.float32)
        b = np.array([x2 - x1, y2 - y1], dtype=np.float32)
        det = np.linalg.det(A)
        if abs(det) < 1e-8:
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        t1, t2 = np.linalg.solve(A, b)
        xi = x1 + t1 * vx1
        yi = y1 + t1 * vy1
        return float(xi), float(yi)


def transform_points_affine(points, M):
    """
    Transform Nx2 points using a 2x3 OpenCV affine matrix.
    """
    points = np.asarray(points, dtype=np.float32)
    ones = np.ones(
        (points.shape[0], 1),
        dtype=np.float32,
    )
    points_h = np.hstack([
        points,
        ones,
    ])
    transformed = points_h @ M.T
    return transformed


def save_image(path, image):
    """
    Save image correctly, including alpha-channel TIFFs,
    using lossless TIFF compression.
    """
    global config
    pil_image_suffix = ("." + config.pil_image_save_kwargs["format"].lower())
    assert Path(path).suffix == pil_image_suffix, f"bad path: {path!r}"
    path = str(path)
    if image.ndim == 2:
        # Grayscale
        pil_image = PIL.Image.fromarray(image)
    elif image.shape[2] == 3:
        # OpenCV BGR -> RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(rgb)
    elif image.shape[2] == 4:
        # OpenCV BGRA -> RGBA
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        pil_image = PIL.Image.fromarray(rgba)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    pil_image.save(path, **config.pil_image_save_kwargs)


def edge_is_white(
    image,
    edge,
    inner_depth=WHITE_TEST_INNER,
    outer_depth=WHITE_TEST_OUTER,
    white_threshold=PRACTICALLY_WHITE_THRESHOLD,
    white_fraction_threshold=WHITE_FRACTION_THRESHOLD,
):
    """
    Return True if the region 10..20 pixels inside an edge is
    practically white, indicating that page content probably
    does not reach the page edge.

    The test is based on the fraction of practically-white pixels,
    rather than average brightness, so a small amount of dark content
    is not hidden by averaging.
    """

    h, w = image.shape[:2]

    # Convert to grayscale for the lightness test.
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    # note: we exclude corners with `inner_depth:-inner_depth`

    # Make sure the requested strip fits.
    if edge == "top":
        if outer_depth > h:
            return False
        # strip = gray[inner_depth:outer_depth, :]
        strip = gray[inner_depth:outer_depth, inner_depth:-inner_depth]

    elif edge == "bottom":
        if outer_depth > h:
            return False
        # strip = gray[h - outer_depth:h - inner_depth, :]
        strip = gray[h - outer_depth:h - inner_depth, inner_depth:-inner_depth]

    elif edge == "left":
        if outer_depth > w:
            return False
        # strip = gray[:, inner_depth:outer_depth]
        strip = gray[inner_depth:-inner_depth, inner_depth:outer_depth]

    elif edge == "right":
        if outer_depth > w:
            return False
        # strip = gray[:, w - outer_depth:w - inner_depth]
        strip = gray[inner_depth:-inner_depth, w - outer_depth:w - inner_depth]

    else:
        raise ValueError(f"Invalid edge: {edge}")

    if strip.size == 0:
        return False

    practically_white = strip >= white_threshold
    white_fraction = np.mean(practically_white)

    if DEBUG:
        print(f"white_fraction: {white_fraction}")

    return white_fraction >= white_fraction_threshold


def paint_white_edge_border(
    image,
    edge,
    border_width=WHITE_BORDER_WIDTH,
):
    """
    Paint a white border inside one edge of the image.
    """

    if edge == "top":
        image[:border_width, :, :] = 255

    elif edge == "bottom":
        image[-border_width:, :, :] = 255

    elif edge == "left":
        image[:, :border_width, :] = 255

    elif edge == "right":
        image[:, -border_width:, :] = 255

    else:
        raise ValueError(f"Invalid edge: {edge}")


def process_image(in_path, out_path):
    """
    Robust page extraction that handles missing top (or bottom) edges.
    - preserves original colors (warps the original image)
    - builds orthogonal axes (no shear)
    - if top is missing, uses bottom + good vertical to infer top by moving up expected_h
    - fills outside the filled page polygon with pure white
    - extensive debug output in OUTPUT_DIR/debug/<page_num>/
    """
    # ---------- start ----------

    # 1. load image

    fname = in_path.name
    m = re.match(r"^(\d+)", fname)
    page_num = int(m.group(1)) if m else 0
    bad_on_left = (page_num % 2 == 1)

    img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to read", in_path); return
    input_is_grayscale = (len(img.shape) == 2)
    if len(img.shape) == 2:
        # OpenCV expects 3-channel images for many operations
        # later convert back to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    H_img, W_img = img.shape[:2]

    # fixed
    # # FIXME wrong H_img?
    # expected_w = int(round(ASPECT * H_img))
    # expected_h = H_img

    # dbgdir = OUTPUT_DIR.joinpath("debug", f"{page_num:03d}")
    # if DEBUG: ensure_dir(dbgdir)



    # 2. calculate search margins

    # Compute the expected ranges of margins
    # NOTE the scanner removes the "scan top" margin
    # so in the X direction, we have only one margin
    # so here we divide by 4, not by 2
    edge_search_width_mm = (
        config.rotated_margined_scan_x -
        config.rotated_scan_x
    ) / 4.0
    edge_search_height_mm = (
        config.rotated_margined_scan_y -
        config.rotated_scan_y
    ) / 2.0

    if DEBUG:
        print(f"config.rotated_margined_scan: ({config.rotated_margined_scan_x}, {config.rotated_margined_scan_y}) mm")
        print(f"config.rotated_scan: ({config.rotated_scan_x}, {config.rotated_scan_y}) mm")
        print(f"margin_range: ({edge_search_width_mm}, {edge_search_height_mm}) mm")

    # Convert them to pixels
    edge_search_width_px = px_of_mm(
        edge_search_width_mm,
        config.scan_resolution
    )
    edge_search_height_px = px_of_mm(
        edge_search_height_mm,
        config.scan_resolution
    )

    # TODO rename, move to config

    SEARCH_MARGIN_FACTOR = 2.0
    SEARCH_MARGIN_ADD_MM = 5

    SEARCH_MARGIN_FACTOR = 1.2
    SEARCH_MARGIN_ADD_MM = 2

    # debug: dont increase
    # SEARCH_MARGIN_FACTOR = 1; SEARCH_MARGIN_ADD_MM = 0

    # Then enlarge them
    edge_search_width_px *= SEARCH_MARGIN_FACTOR
    edge_search_height_px *= SEARCH_MARGIN_FACTOR
    search_margin_add_px = px_of_mm(
        SEARCH_MARGIN_ADD_MM,
        config.scan_resolution
    )
    edge_search_width_px += search_margin_add_px
    edge_search_height_px += search_margin_add_px

    # we need integers for array indices
    edge_search_width_px = int(math.ceil(edge_search_width_px))
    edge_search_height_px = int(math.ceil(edge_search_height_px))



    # 3. build mask

    # Step 1: Segment page vs. gray background
    # -> gray, mask

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu is usually sufficient here
    _, page_mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Page should be white
    if np.mean(gray[page_mask == 255]) < np.mean(gray[page_mask == 0]):
        page_mask = cv2.bitwise_not(page_mask)

    page_mask = cv2.morphologyEx(
        page_mask,
        cv2.MORPH_CLOSE,
        np.ones((5,5), np.uint8)
    )



    # TODO rename
    mask = page_mask



    # Step 2: Scan for transitions
    # Step 3: Transition verification

    # NOTE we ignore inside_pts because
    # - guillotine cutter can produce non-orthogonal inside edge of the book
    # - document scanner does remove the top edge of the scan
    # if config.do_rotate is True then
    # "inside edge of the book" == "top edge of the scan"
    #
    # this also means:
    # the widths of the top and bottom edges
    # must have no influence on the deskew step
    # (deskew = Rotation + Perspective transform)

    if bad_on_left:
        outside_pts = detect_edge_points(
            gray,
            mask,
            EDGE_RIGHT,
            edge_search_width_px,
        )
    else:
        outside_pts = detect_edge_points(
            gray,
            mask,
            EDGE_LEFT,
            edge_search_width_px,
        )

    top_pts = detect_edge_points(
        gray,
        mask,
        EDGE_TOP,
        edge_search_height_px,
    )

    bottom_pts = detect_edge_points(
        gray,
        mask,
        EDGE_BOTTOM,
        edge_search_height_px,
    )



    # 4. reject outliers

    # Step 4: Reject isolated outliers
    # Use a median filter before RANSAC.

    top_pts = reject_outliers_horizontal(top_pts)
    bottom_pts = reject_outliers_horizontal(bottom_pts)
    outside_pts = reject_outliers_vertical(outside_pts)



    # 5. fit lines

    # Step 5: RANSAC

    top_line = fit_line_ransac(top_pts)[:4]
    bottom_line = fit_line_ransac(bottom_pts)[:4]
    outside_line = fit_line_ransac(outside_pts)[:4]

    if 0:
        # debug
        # float(vx), float(vy), float(x0), float(y0)
        print(f"top_line={top_line}")
        print(f"bottom_line={bottom_line}")
        print(f"outside_line={outside_line}")



    # page margin
    outside_top = intersect_lines(
        outside_line,
        top_line
    )
    outside_bottom = intersect_lines(
        outside_line,
        bottom_line
    )

    # also calculate inside_top and inside_bottom
    # inside_line is simply the inside edge of the source image
    # y1 = 0; y2 = y_max
    # on odd pages: x1 = x2 = 0 # inside is left
    # on even pages: x1 = x2 = x_max # inside is right
    if bad_on_left:
        # Odd page:
        # binding/inside edge is the LEFT image boundary
        inside_x = 0.0
    else:
        # Even page:
        # binding/inside edge is the RIGHT image boundary
        inside_x = float(W_img - 1)
    inside_top = intersect_line_with_vertical_boundary(
        top_line,
        inside_x
    )
    inside_bottom = intersect_line_with_vertical_boundary(
        bottom_line,
        inside_x
    )



    if DEBUG:
        vis = img.copy()
        # margin range
        if bad_on_left:
            # outside edge is right
            # no line on the left
            if 0:
                # Wr, Hr
                pts = np.array([
                    [0, edge_search_height_px], # top left
                    [Wr - edge_search_width_px, edge_search_height_px], # top right
                    [Wr - edge_search_width_px, Hr - edge_search_height_px], # bottom right
                    [0, Hr - edge_search_height_px], # bottom left
                ], np.int32)
            else:
                # W_img, H_img
                pts = np.array([
                    [0, edge_search_height_px], # top left
                    [W_img - edge_search_width_px, edge_search_height_px], # top right
                    [W_img - edge_search_width_px, H_img - edge_search_height_px], # bottom right
                    [0, H_img - edge_search_height_px], # bottom left
                ], np.int32)
        else:
            # outside edge is left
            # no line on the right
            if 0:
                # Wr, Hr
                pts = np.array([
                    [Wr, edge_search_height_px], # top right
                    [edge_search_width_px, edge_search_height_px], # top left
                    [edge_search_width_px, Hr - edge_search_height_px], # bottom left
                    [Wr, Hr - edge_search_height_px], # bottom right
                ], np.int32)
            else:
                # W_img, H_img
                pts = np.array([
                    [W_img, edge_search_height_px], # top right
                    [edge_search_width_px, edge_search_height_px], # top left
                    [edge_search_width_px, H_img - edge_search_height_px], # bottom left
                    [W_img, H_img - edge_search_height_px], # bottom right
                ], np.int32)
        cv2.polylines(vis, [pts], False, (0,255,0), 3) # green

        # Margin range
        if 0:
            # expand the binding edge to expected_w
            if bad_on_left:
                # outside edge is the RIGHT edge
                x1_top = outside_top[0]
                x1_bottom = outside_bottom[0]
                x0_top = x1_top - expected_w
                x0_bottom = x1_bottom - expected_w
            else:
                # outside edge is the LEFT edge
                x0_top = outside_top[0]
                x0_bottom = outside_bottom[0]
                x1_top = x0_top + expected_w
                x1_bottom = x0_bottom + expected_w
        else:
            # dont expand the binding edge to expected_w
            # use only the detected page edges
            if bad_on_left:
                # outside edge is right
                x1_top = outside_top[0]
                x1_bottom = outside_bottom[0]
                # binding edge
                # find leftmost detected page boundary
                # x0_top = np.min(top_pts[:,0])
                # x0_bottom = np.min(bottom_pts[:,0])
                x0_top = 0
                x0_bottom = 0
            else:
                # outside edge is left
                x0_top = outside_top[0]
                x0_bottom = outside_bottom[0]
                # binding edge
                # find rightmost detected page boundary
                # x1_top = np.max(top_pts[:,0])
                # x1_bottom = np.max(bottom_pts[:,0])
                if 0:
                    # Wr, Hr
                    x1_top = Wr - 1
                    x1_bottom = Wr - 1
                else:
                    # W_img, H_img
                    x1_top = W_img - 1
                    x1_bottom = W_img - 1

        # Detected page quadrilateral
        if 0:
            pts = np.array([
                # top left: bad: y=outside_top[1] is wrong
                # this would have to be min(top_line) or so
                [x0_top, outside_top[1]], # top left
                [x1_top, outside_top[1]], # top right: good
                [x1_bottom, outside_bottom[1]], # bottom right: good
                [x0_bottom, outside_bottom[1]], # bottom left: bad: y=outside_bottom[1] is wrong
            ], np.int32)
        elif 0:
            pts = np.array([
                [x0_top, outside_top[1]],
                [x1_top, outside_top[1]],
                [x1_bottom, outside_bottom[1]],
                [x0_bottom, outside_bottom[1]],
            ], np.int32)
        elif 1:
            pts = np.round(np.array([
                inside_top,
                outside_top,
                outside_bottom,
                inside_bottom,
            ], dtype=np.float32)).astype(np.int32)
        cv2.polylines(vis, [pts], True, (0,0,255), 3) # red

        name = OUTPUT_DIR / f"{page_num:03d}.line-0900-fit-lines-ransac.tiff"
        # print(f"writing {name}")
        save_image(name, vis)

    src_corners = np.float32([
        inside_top,
        outside_top,
        outside_bottom,
        inside_bottom,
    ])



    # FIXME merge old code with new code



    # # TODO remove

    # gray, mask, contours = get_gray_mask_contours(img, dbgdir)

    # if not contours:
    #     print(f"Warning: no contours found in {in_path}")
    #     return

    # page_contour = max(contours, key=cv2.contourArea)

    if config.use_three_edge_deskew:

        # top_pts, bottom_pts, outside_pts = split_edge_candidates(
        #     page_contour,
        #     bad_on_left
        # )

        # top_line = fit_line_ransac(top_pts)[:4]
        # bottom_line = fit_line_ransac(bottom_pts)[:4]
        # outside_line = fit_line_ransac(outside_pts)[:4]



        # 6. rotate
        # deskew part 1

        # old
        # top_angle = math.degrees(line_angle(top_line))
        # bottom_angle = math.degrees(line_angle(bottom_line))
        # outside_angle = math.degrees(line_angle(outside_line))

        # new
        top_angle = horizontal_line_angle(top_line)
        bottom_angle = horizontal_line_angle(bottom_line)
        outside_angle = vertical_line_angle(outside_line)

        if DEBUG:
            # start debug prints
            print()
            print(f"line 570: page_num={page_num}")

        rotation_error = np.mean([
            top_angle,
            bottom_angle,
            outside_angle - 90,
        ])

        Mrot = cv2.getRotationMatrix2D(
            (W_img/2, H_img/2),
            rotation_error,
            1.0
        )

        if DEBUG:
            print(
                f"line 575: before rotation: "
                f"top_angle={top_angle:.3f} "
                f"bottom_angle={bottom_angle:.3f} "
                f"outside_angle={outside_angle:.3f}"
            )

        if remove_inside_transparent_strip:
            # remove the extra inside edge
            # rotate with gray background
            rotated = cv2.warpAffine(
                img,
                Mrot,
                (W_img, H_img),
                # borderValue=(255,255,255) # white
                borderValue=(128,128,128) # gray
            )
        else:
            # keep the extra inside edge
            # rotate with transparent background
            # add alpha channel
            img_bgra = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2BGRA,
            )
            # init alpha channel
            # img_bgra[:, :, 3] = 255
            rotated_bgra = cv2.warpAffine(
                img_bgra,
                Mrot,
                (W_img, H_img),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
            rotated = rotated_bgra

        Hr, Wr = rotated.shape[:2]



        # also rotate the detected edges

        rotated_corners = transform_points_affine(
            src_corners,
            Mrot,
        )

        inside_top = rotated_corners[0]
        outside_top = rotated_corners[1]
        outside_bottom = rotated_corners[2]
        inside_bottom = rotated_corners[3]

        # remember the inside edge for later cropping
        actual_inside_top = rotated_corners[0].copy()
        actual_inside_bottom = rotated_corners[3].copy()

        # ignore the inside edge for perspective transform
        # Ignore the unknown inside-edge angle for perspective correction.
        # Use the image boundary as the inside edge.
        if bad_on_left:
            # outside edge is the RIGHT edge
            # inside edge is the LEFT edge
            inside_top[0] = 0
            inside_bottom[0] = 0
        else:
            # outside edge is the LEFT edge
            # inside edge is the RIGHT edge
            inside_top[0] = Wr - 1
            inside_bottom[0] = Wr - 1

        top_width = np.linalg.norm(
            outside_top - inside_top
        )
        bottom_width = np.linalg.norm(
            outside_bottom - inside_bottom
        )
        left_height = np.linalg.norm(
            inside_bottom - inside_top
        )
        right_height = np.linalg.norm(
            outside_bottom - outside_top
        )

        # expected size after rotation
        expected_w = int(round(
            (top_width + bottom_width) / 2.0
        ))
        expected_h = int(round(
            (left_height + right_height) / 2.0
        ))
        if 0:
            # Or, if you want the height to be determined specifically by the outside edge
            # The average is usually more stable, though.
            expected_h = int(round(
                right_height
            ))

        if DEBUG:
            print("line 580: rotated size", Wr, Hr)

        # img = rotated # ?



        if 0:
            # approximated page height
            page_height = math.dist(
                outside_top,
                outside_bottom
            )
        else:
            # perpendicular page height
            page_height = math.hypot(
                outside_bottom[0] - outside_top[0],
                outside_bottom[1] - outside_top[1]
            )

        # # no. this fails to reconstruct the page height...
        # # TODO try to solve this with the average height of multiple pages
        # # assuming all pages must have the same height
        # # also allowing the user to specify a scale_y factor
        # if 0:
        #     if config.do_rotate == False:
        #         # fix scan height
        #         # ...
        #         pass

        expected_h = int(round(page_height))

        # # ASPECT = x / y
        # # x = y * ASPECT

        # # expand the binding edge to expected_w
        # # expected_w = int(round(expected_h * ASPECT))
        # expected_w = int(round(page_height * ASPECT))

        if DEBUG:
            vis = rotated.copy()
            pts_rotated = np.round(
                rotated_corners
            ).astype(np.int32)
            cv2.polylines(
                vis,
                [pts_rotated],
                True,
                (0, 0, 255), # red
                3,
            )
            name = OUTPUT_DIR / (f"{page_num:03d}.line-1200-rotated-corners.tiff")
            save_image(str(name), vis)



        # 7b. cleanup the inside edge
        # use outmost_inside_x to remove the half-transparent inside edge
        # if remove_inside_transparent_strip:
        if 0:
            if bad_on_left:
                # odd page
                # Inside edge is on the LEFT.
                # Remove everything from x=0 through outmost_inside_x.
                crop_x = int(math.ceil(outmost_inside_x))
                rotated = rotated[:, crop_x:, :]

            else:
                # even page
                # Inside edge is on the RIGHT.
                # Remove everything from outmost_inside_x through the right edge.
                crop_x = int(math.floor(outmost_inside_x))
                rotated = rotated[:, :crop_x, :]



        # 8. perspective transform
        # deskew part 2

        # crop as a quadrilateral
        # not better than "crop as a rectangle"?
        if 0:
            # expand the binding edge to expected_w
            if bad_on_left:
                # outside edge is the RIGHT edge
                x1_top = outside_top[0]
                x1_bottom = outside_bottom[0]
                x0_top = x1_top - expected_w
                x0_bottom = x1_bottom - expected_w
            else:
                # outside edge is the LEFT edge
                x0_top = outside_top[0]
                x0_bottom = outside_bottom[0]
                x1_top = x0_top + expected_w
                x1_bottom = x0_bottom + expected_w
        else:
            # dont expand the binding edge to expected_w
            # use only the detected page edges
            # if bad_on_left:
            #     # outside edge is RIGHT edge
            #     x1_top = outside_top[0]
            #     x1_bottom = outside_bottom[0]
            #     # use the actual detected left edge
            #     x0_top = np.min(page_contour[:,0,0])
            #     x0_bottom = x0_top
            # else:
            #     # outside edge is LEFT edge
            #     x0_top = outside_top[0]
            #     x0_bottom = outside_bottom[0]
            #     # use the actual detected right edge
            #     x1_top = np.max(page_contour[:,0,0])
            #     x1_bottom = x1_top
            # problem: page_contour after thresholding may include the background
            # or may not have a reliable missing-edge position.
            # A cleaner temporary solution is to use the detected quadrilateral width
            # from the two horizontal edge intersections
            if bad_on_left:
                # outside edge is right
                x1_top = outside_top[0]
                x1_bottom = outside_bottom[0]
                # binding edge
                # find leftmost detected page boundary
                # x0_top = np.min(top_pts[:,0])
                # x0_bottom = np.min(bottom_pts[:,0])
                x0_top = 0
                x0_bottom = 0
            else:
                # outside edge is left
                x0_top = outside_top[0]
                x0_bottom = outside_bottom[0]
                # binding edge
                # find rightmost detected page boundary
                # x1_top = np.max(top_pts[:,0])
                # x1_bottom = np.max(bottom_pts[:,0])
                x1_top = Wr - 1
                x1_bottom = Wr - 1

            # dont expand the binding edge to expected_w
            expected_w = int(round(
                math.dist((x0_top, outside_top[1]), (x1_top, outside_top[1]))
            ))

        if bad_on_left:
            # inside = LEFT
            # outside = RIGHT
            src = np.float32([
                inside_top,       # TL
                outside_top,      # TR
                outside_bottom,   # BR
                inside_bottom,    # BL
            ])
        else:
            # outside = LEFT
            # inside = RIGHT
            src = np.float32([
                outside_top,      # TL
                inside_top,       # TR
                inside_bottom,    # BR
                outside_bottom,   # BL
            ])

        dst = np.float32([
            [0, 0],
            [expected_w - 1, 0],
            [expected_w - 1, expected_h - 1],
            [0, expected_h - 1],
        ])

        M = cv2.getPerspectiveTransform(src, dst)

        crop = cv2.warpPerspective(
            rotated,
            M,
            (expected_w, expected_h),
            borderValue=(255,255,255)
        )

        if DEBUG:
            print(
                "line 650: crop",
                f"W_img={W_img}",
                f"H_img={H_img}",
                f"outside_top={outside_top}",
                f"outside_bottom={outside_bottom}",
                f"x1_top={x1_top}",
                f"x1_bottom={x1_bottom}",
                f"x0_top={x0_top}",
                f"x0_bottom={x0_bottom}",
                f"expected_w={expected_w}",
                f"expected_h={expected_h}",
            )

        actual_inside_warped = cv2.perspectiveTransform(
            np.array([[
                actual_inside_top,
                actual_inside_bottom,
            ]], dtype=np.float32),
            M,
        )[0]

        actual_inside_top_warped = actual_inside_warped[0]
        actual_inside_bottom_warped = actual_inside_warped[1]

        inside_top_x = actual_inside_top_warped[0]
        inside_bottom_x = actual_inside_bottom_warped[0]
        if bad_on_left:
            # odd page
            # inside edge is the LEFT edge
            outmost_inside_x = max(inside_top_x, inside_bottom_x)
        else:
            # even page
            # inside edge is the RIGHT edge
            outmost_inside_x = min(inside_top_x, inside_bottom_x)

        # crop = rotated[
        #     0:expected_h,
        #     int(x0_page):int(x1_page)
        # ]

        # y_top = round(outside_top[1])
        # y_bottom = round(outside_bottom[1])
        # actual_height = y_bottom - y_top
        # crop = rotated[
        #     y_top:y_top + expected_h,
        #     x0_page:x1_page
        # ]

        # y_top = int(round(outside_top[1]))
        # y_bottom = int(round(outside_bottom[1]))
        # crop = rotated[
        #     y_top:y_bottom,
        #     x0_page:x1_page
        # ]

        if 0:
            # img = repair_binding(img, bad_on_left, width=50)
            if 1:
                # rotated = repair_binding(rotated, bad_on_left, width=50)
                # Hr, Wr = rotated.shape[:2]
                # print("line 660: rotated size", Wr, Hr)
                # warped = rotated
                crop = repair_binding(crop, bad_on_left, width=50)
                Hr, Wr = crop.shape[:2]
                if DEBUG:
                    print("line 660: crop size", Wr, Hr)
            else:
                crop = repair_binding(crop, bad_on_left, width=50)

            if DEBUG:
                print(
                    "line 760: after repair_binding: crop actual:",
                    f"crop.shape[1]={crop.shape[1]}",
                    f"expected_w={expected_w}",
                )

        warped = crop



        # 8b. cleanup the inside edge
        # use outmost_inside_x to remove the half-transparent inside edge
        if remove_inside_transparent_strip:
            if bad_on_left:
                # Inside edge is on the LEFT.
                # Remove everything from x=0 through outmost_inside_x.
                crop_x = int(math.ceil(outmost_inside_x))
                warped = warped[:, crop_x:, :]

            else:
                # Inside edge is on the RIGHT.
                # Remove everything from outmost_inside_x through the right edge.
                crop_x = int(math.floor(outmost_inside_x))
                warped = warped[:, :crop_x, :]



        # 8c. remove small gray artifacts along page edges
        #
        # The perspective transform can leave thin light-gray lines
        # along the page edges because the detected page boundary
        # is not perfectly straight.
        #
        # However, do NOT paint over page content in borderless printing.
        #
        # For each edge:
        #   - inspect a strip 20..10 pixels inside the page
        #   - if >= 99% of pixels are practically white,
        #     assume there is no content near that edge
        #   - paint a 10-pixel white border inside that edge
        #
        # The decision is made independently for each edge.

        # Apply independently to all four edges.
        for edge in ("top", "bottom", "left", "right"):

            if not edge_is_white(warped, edge):
                # Content reaches the page edge.
                # Do not destroy it.
                if DEBUG:
                    print(f"white edge cleanup: {edge}: borderless -> KEEP CONTENT")
            else:
                # The 10..20 px interior strip is practically white,
                # so it is safe to remove the gray artifact at the edge.
                paint_white_edge_border(warped, edge, WHITE_BORDER_WIDTH)

                if DEBUG:
                    print(f"white edge cleanup: {edge}: white margin -> PAINT {WHITE_BORDER_WIDTH}px")



    else:
        # config.use_three_edge_deskew == False

        raise NotImplementedError("sorry, your book is too tall for your scanner...")

        # FIXME use only two page edges: outside, bottom

        # # Approximate contour to quadrilateral
        # epsilon = 0.02 * cv2.arcLength(page_contour, True)
        # approx = cv2.approxPolyDP(page_contour, epsilon, True)
        # if len(approx) != 4:
        #     approx = cv2.convexHull(page_contour)
        #     if len(approx) < 4:
        #         print(f"Warning: not enough points for perspective in {in_path}")
        #         return
        #     # pick 4 extreme points
        #     pts = np.array([
        #         approx[approx[:,0,0].argmin()][0],  # leftmost
        #         approx[approx[:,0,1].argmin()][0],  # topmost
        #         approx[approx[:,0,0].argmax()][0],  # rightmost
        #         approx[approx[:,0,1].argmax()][0]   # bottommost
        #     ])
        # else:
        #     pts = approx.reshape(4,2)

        rect = order_points(pts)

        # Perspective transform
        widthA = np.linalg.norm(rect[2] - rect[3])
        widthB = np.linalg.norm(rect[1] - rect[0])
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.linalg.norm(rect[1] - rect[2])
        heightB = np.linalg.norm(rect[0] - rect[3])
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # # Add internal white border
    # h, w = warped.shape[:2]
    # canvas = np.ones_like(warped) * 255
    # b = BORDER_SIZE
    # canvas[b:h-b, b:w-b] = warped[b:h-b, b:w-b]

    h, w = warped.shape[:2]
    b = BORDER_SIZE

    # Prepare a canvas with the same content as warped
    canvas = warped.copy()

    # Function to compute average color along a strip
    def avg_color_strip(img, axis, start, end, strip_width=1):
        if axis == 'top':
            strip = img[start:end, :, :]
        elif axis == 'bottom':
            strip = img[h-end:h-start, :, :]
        elif axis == 'left':
            strip = img[:, start:end, :]
        elif axis == 'right':
            strip = img[:, w-end:w-start, :]
        else:
            raise ValueError("Invalid axis")
        return np.mean(strip, axis=(0,1)).astype(np.uint8)

    # FIXME preserve patterns near edges
    if 0:
        # Fill borders with local average color
        canvas[0:b, :, :] = avg_color_strip(canvas, 'top', 50, 100)       # top border
        canvas[h-b:h, :, :] = avg_color_strip(canvas, 'bottom', 50, 100)  # bottom border
        canvas[:, 0:b, :] = avg_color_strip(canvas, 'left', 50, 100)      # left border
        canvas[:, w-b:w, :] = avg_color_strip(canvas, 'right', 50, 100)   # right border



    # 9. save

    if input_is_grayscale:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    ensure_dir(out_path.parent)
    # cv2.imwrite(out_path, canvas, config.cv2_imwrite_params)
    save_image(out_path, canvas)
    # print(f"writing {out_path}")


def _process_file(in_path):
    # page_num = get_page_num(in_path)

    out_path = OUTPUT_DIR / in_path.name

    process_image(in_path, out_path)


def main():
    ensure_dir(OUTPUT_DIR)

    files = sorted(INPUT_DIR.glob(f"*.{config.scan_format}"))

    if not files:
        print("No image files found in", INPUT_DIR)
        return

    # Only submit work for files that still need processing.
    files = remove_done_files(files, OUTPUT_DIR)

    if 0:
        # debug: process only some pages
        def filter_file(file):
            page_num = get_page_num(file)
            if not page_num in (1, 2, 3):
                return False
            return True
        files = list(filter(filter_file, files))

    if not files:
        print("nothing to do")
        return

    dst = OUTPUT_DIR
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

    num_workers = psutil.cpu_count(logical=False) or 1

    # by default, OpenCV uses multiple CPU threads
    cv2.setNumThreads(1)

    if 0:
        # debug: disable parallel processing
        num_workers = 1

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_file, fname): fname
            for fname in files
        }

        tqdm_kwargs = dict(
            total=len(files),
            ncols=80,
            unit="page",
        )

        with tqdm(**tqdm_kwargs) as pbar:
            for future in as_completed(futures):
                fname = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"Error processing {fname}: {exc}")
                    raise
                finally:
                    pbar.update(1)


if __name__ == "__main__":
    main()
