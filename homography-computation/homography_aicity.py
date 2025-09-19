import json, cv2 as cv, numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
SCENE_DIR = ROOT / "PhysicalAI-SmartSpaces_py" / "MTMC_Tracking_2025" / "train" / "Warehouse_012"
CALIB = SCENE_DIR / "calibration.json"
MAP_PNG = SCENE_DIR / "map.png"                 # dùng map gốc
CAM_ID  = "Camera_00"
VIDEO   = SCENE_DIR / "videos" / f"{CAM_ID}.mp4"

# === bù định hướng map (theo scene này): xoay 180° + flip dọc ===
ROT_DEG = 180
FLIP_X  = True    # mirror theo trục dọc (left–right)
FLIP_Y  = False
# ===============================================================

def get_cam(calib, cam_id):
    for s in calib["sensors"]:
        if s.get("id") == cam_id:
            return s
    raise KeyError(f"Không thấy {cam_id}")

def normalizeH(H):
    return H / H[2,2] if abs(H[2,2])>1e-12 else H

def T_XY_to_pix_B(scale, txy):
    # Quy ước B: y_pixel tăng LÊN (đã kiểm chứng đúng với bộ này)
    s, tx, ty = float(scale), float(txy["x"]), float(txy["y"])
    return np.array([[ s, 0.,  s*tx],
                     [ 0., s,  s*ty],
                     [ 0., 0.,  1.  ]], dtype=np.float64)

def rot_center_3x3(w, h, deg):
    M2 = cv.getRotationMatrix2D((w/2.0, h/2.0), deg, 1.0)
    M3 = np.eye(3, dtype=np.float64); M3[:2,:] = M2; return M3

def flip_center_3x3(w, h, fx=False, fy=False):
    sx, sy = (-1 if fx else 1), (-1 if fy else 1)
    cx, cy = w/2.0, h/2.0
    T1 = np.array([[1,0,cx],[0,1,cy],[0,0,1]], dtype=np.float64)
    D  = np.array([[sx,0,0],[0,sy,0],[0,0,1]], dtype=np.float64)
    T2 = np.array([[1,0,-cx],[0,1,-cy],[0,0,1]], dtype=np.float64)
    return T1 @ D @ T2

def merge(dst, warped):
    gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    mask = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)[1]
    inv  = cv.bitwise_not(mask)
    base = cv.bitwise_and(dst, dst, mask=inv)
    over = cv.bitwise_and(warped, warped, mask=mask)
    return cv.add(base, over)

def show_half(win, img):
    h, w = img.shape[:2]
    cv.imshow(win, cv.resize(img, (w//2, h//2)))

def main():
    map_img = cv.imread(str(MAP_PNG), cv.IMREAD_COLOR)
    if map_img is None: raise FileNotFoundError("Không đọc được map.png")
    mh, mw = map_img.shape[:2]

    calib = json.load(open(CALIB, "r", encoding="utf-8"))
    cam = get_cam(calib, CAM_ID)

    # H: XY -> img, đảo để ảnh -> XY
    H = normalizeH(np.array(cam["homography"], dtype=np.float64))
    H_inv = np.linalg.inv(H)

    # XY -> map pixel (B)
    T_B = T_XY_to_pix_B(cam["scaleFactor"], cam["translationToGlobalCoordinates"])

    # Bù định hướng map: xoay 180° + flip dọc quanh TÂM map
    S_orient = rot_center_3x3(mw, mh, ROT_DEG) @ flip_center_3x3(mw, mh, FLIP_X, FLIP_Y)

    # Ảnh -> map pixel
    H_img2map = normalizeH(S_orient @ T_B @ H_inv)

    # lấy frame đầu, warp + merge
    cap = cv.VideoCapture(str(VIDEO)); ok, frame = cap.read(); cap.release()
    if not ok: raise RuntimeError(f"Không đọc được frame từ {VIDEO}")
    cv.imshow(f"{CAM_ID} raw (1/2)", cv.resize(frame, (frame.shape[1]//2, frame.shape[0]//2)))

    warped = cv.warpPerspective(frame, H_img2map, (mw, mh), flags=cv.INTER_LINEAR)
    merged = merge(map_img, warped)

    show_half("map (1/2)", map_img)

    note = f"ROT={ROT_DEG} | FLIP_X={FLIP_X} FLIP_Y={FLIP_Y} | conv=B"
    vis = merged.copy(); cv.putText(vis, note, (20,40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv.LINE_AA)
    show_half("merged+state (1/2)", vis)

    cv.waitKey(0); cv.destroyAllWindows()

if __name__ == "__main__":
    main()
