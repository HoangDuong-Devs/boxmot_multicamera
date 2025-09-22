import json
import numpy as np
import cv2 as cv

# ==== Config ====
SRC_IMG = 'imgs/src.jpg'
DST_IMG = 'imgs/dst.jpg'
POINTS_JSON = 'points.json'
DISPLAY_MAX_W = 900      # thu nhỏ ảnh khi preview cho đỡ to
RANSAC_THRESH = 5.0       # px
ALPHA_INIT = 70           # 0..100

# ==== Globals ====
src_pts, dst_pts = [], []
src_img = cv.imread(SRC_IMG, cv.IMREAD_COLOR)
dst_img = cv.imread(DST_IMG, cv.IMREAD_COLOR)
if src_img is None or dst_img is None:
    raise FileNotFoundError("Không đọc được ảnh SRC/DST")
src_vis = src_img.copy()
dst_vis = dst_img.copy()
H = None
alpha = ALPHA_INIT

# scale preview (không làm thay đổi toạ độ thật)
def resize_for_display(img, max_w=DISPLAY_MAX_W):
    h, w = img.shape[:2]
    if w <= max_w:
        return img, 1.0
    scale = max_w / w
    return cv.resize(img, (int(w*scale), int(h*scale))), scale

src_disp, s_scale = resize_for_display(src_vis)
dst_disp, d_scale = resize_for_display(dst_vis)

def draw_points():
    global src_vis, dst_vis, src_disp, dst_disp
    src_vis = src_img.copy()
    dst_vis = dst_img.copy()
    for (x, y) in src_pts:
        cv.circle(src_vis, (int(x), int(y)), 5, (0,0,255), -1)
    for (x, y) in dst_pts:
        cv.circle(dst_vis, (int(x), int(y)), 5, (0,0,255), -1)
    src_disp, _ = resize_for_display(src_vis)
    dst_disp, _ = resize_for_display(dst_vis)

def on_mouse_src(event, x, y, flags, param):
    # map từ disp coords -> original coords
    X = int(x / s_scale); Y = int(y / s_scale)
    if event == cv.EVENT_LBUTTONDOWN:
        src_pts.append((X, Y))
        draw_points()
    elif event == cv.EVENT_RBUTTONDOWN:
        if src_pts: src_pts.pop()
        draw_points()

def on_mouse_dst(event, x, y, flags, param):
    X = int(x / d_scale); Y = int(y / d_scale)
    if event == cv.EVENT_LBUTTONDOWN:
        dst_pts.append((X, Y))
        draw_points()
    elif event == cv.EVENT_RBUTTONDOWN:
        if dst_pts: dst_pts.pop()
        draw_points()

def compute_h():
    global H
    if len(src_pts) < 4 or len(dst_pts) < 4 or len(src_pts) != len(dst_pts):
        print("[!] Cần ≥4 cặp điểm và số lượng hai phía phải bằng nhau.")
        return None
    src = np.array(src_pts, dtype=np.float32).reshape(-1,1,2)
    dst = np.array(dst_pts, dtype=np.float32).reshape(-1,1,2)
    H, mask = cv.findHomography(src, dst, cv.RANSAC, RANSAC_THRESH)
    if H is None:
        print("[!] Không ước lượng được Homography (điểm có thể suy biến/collinear).")
        return None
    inliers = int(mask.sum())
    # Reprojection error (chỉ với inliers)
    src_in = src[mask.ravel()==1]
    dst_in = dst[mask.ravel()==1]
    proj = cv.perspectiveTransform(src_in.reshape(-1,1,2), H)
    err = np.linalg.norm(proj.reshape(-1,2) - dst_in.reshape(-1,2), axis=1)
    print(f"[H] Inliers: {inliers}/{len(src)} | Reproj err (px): mean={err.mean():.2f}, median={np.median(err):.2f}, max={err.max():.2f}")
    print(H)
    return H

def warp_to_dst(h=None):
    if h is None: return None
    h_d, w_d = dst_img.shape[:2]
    warped = cv.warpPerspective(src_img, h, (w_d, h_d))
    return warped

def make_mask(warped):
    # mask = pixel nào có dữ liệu (không hoàn toàn đen)
    gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    mask = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)[1]
    return mask

def blend_over(dst, warped, alpha_ratio=0.7):
    # alpha_ratio in [0,1]
    mask = make_mask(warped)
    mask_3 = cv.merge([mask, mask, mask])
    fore = cv.addWeighted(warped, alpha_ratio, np.zeros_like(warped), 0, 0)
    # giữ nền dst nơi mask=0, và hoà trộn nơi mask=1
    out = dst.copy()
    out = np.where(mask_3>0, fore + np.where(mask_3>0, dst*(1-alpha_ratio), 0), dst)
    out = out.astype(np.uint8)
    return out

def merge_fast(dst, warped):
    mask = make_mask(warped)
    inv = cv.bitwise_not(mask)
    base = cv.bitwise_and(dst, dst, mask=inv)
    over = cv.bitwise_and(warped, warped, mask=mask)
    return cv.add(base, over)

def on_trackbar(val):
    global alpha
    alpha = val

def save_points():
    data = {"src": src_pts, "dst": dst_pts}
    with open(POINTS_JSON, "w") as f:
        json.dump(data, f)
    print(f"[✓] Saved points -> {POINTS_JSON}")

def load_points():
    global src_pts, dst_pts
    try:
        with open(POINTS_JSON, "r") as f:
            data = json.load(f)
        src_pts = [tuple(map(float, p)) for p in data.get("src", [])]
        dst_pts = [tuple(map(float, p)) for p in data.get("dst", [])]
        draw_points()
        print(f"[✓] Loaded points <- {POINTS_JSON}")
    except Exception as e:
        print(f"[!] Load failed: {e}")

# ==== UI ====
cv.namedWindow('SRC', cv.WINDOW_AUTOSIZE)
cv.namedWindow('DST', cv.WINDOW_AUTOSIZE)
cv.namedWindow('MERGE', cv.WINDOW_AUTOSIZE)
cv.createTrackbar('alpha(%)','MERGE', ALPHA_INIT, 100, on_trackbar)
cv.setMouseCallback('SRC', on_mouse_src)
cv.setMouseCallback('DST', on_mouse_dst)
draw_points()

print("""
Hướng dẫn:
- Trái chuột: thêm điểm   |  Phải chuột: undo
- Cần chấm điểm TƯƠNG ỨNG: SRC và DST phải có cùng số lượng điểm.
- Phím:
  [h]  Tính Homography (RANSAC)
  [w]  Warp SRC -> DST
  [m]  Merge nhanh (che khu vực có dữ liệu)
  [b]  Blend với alpha (dùng trackbar để đổi alpha)
  [p]  Save điểm ra JSON   |  [l] Load điểm từ JSON
  [c]  Clear tất cả điểm   |  [ESC] Thoát
""")

warped_cache = None

while True:
    cv.imshow('SRC', src_disp)
    cv.imshow('DST', dst_disp)

    key = cv.waitKey(10) & 0xFF
    if key == ord('h'):
        H = compute_h()
        warped_cache = None

    elif key == ord('w'):
        if H is None:
            print("[!] Chưa có H. Bấm [h] trước.")
            continue
        warped_cache = warp_to_dst(H)
        if warped_cache is not None:
            cv.imshow('MERGE', warped_cache)

    elif key == ord('m'):
        if H is None:
            print("[!] Chưa có H. Bấm [h] trước.")
            continue
        if warped_cache is None:
            warped_cache = warp_to_dst(H)
        merged = merge_fast(dst_img, warped_cache)
        cv.imshow('MERGE', merged)

    elif key == ord('b'):
        if H is None:
            print("[!] Chưa có H. Bấm [h] trước.")
            continue
        if warped_cache is None:
            warped_cache = warp_to_dst(H)
        a = np.clip(alpha/100.0, 0.0, 1.0)
        blended = blend_over(dst_img, warped_cache, a)
        cv.imshow('MERGE', blended)

    elif key == ord('p'):
        save_points()

    elif key == ord('l'):
        load_points()

    elif key == ord('c'):
        src_pts.clear(); dst_pts.clear()
        H = None; warped_cache = None
        draw_points()
        cv.destroyWindow('MERGE')

    elif key == 27:  # ESC
        break

cv.destroyAllWindows()
