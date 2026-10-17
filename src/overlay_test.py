import json, glob, os
import cv2
import numpy as np

LOBBY = 4.0

COURT = {
    "mid_T_outer": (0, 5),       "mid_T_lobby": (0, LOBBY),
    "mid_B_lobby": (0, -LOBBY),  "mid_B_outer": (0, -5),
    "baulk_L_T": (-3.75, LOBBY), "baulk_L_B": (-3.75, -LOBBY),
    "bonus_L_T": (-4.75, LOBBY), "bonus_L_B": (-4.75, -LOBBY),
    "corner_L_T": (-6.5, 5),     "corner_L_B": (-6.5, -5),
    "baulk_R_T": (3.75, LOBBY),  "baulk_R_B": (3.75, -LOBBY),
    "bonus_R_T": (4.75, LOBBY),  "bonus_R_B": (4.75, -LOBBY),
    "corner_R_T": (6.5, 5),      "corner_R_B": (6.5, -5),
}

# (start, end, colour BGR)
LINES = [
    ((0, -5), (0, 5), (0, 255, 0)),            # midline
    ((-3.75, -LOBBY), (-3.75, LOBBY), (255, 255, 0)),  # baulk L
    ((3.75, -LOBBY), (3.75, LOBBY), (255, 255, 0)),    # baulk R
    ((-4.75, -LOBBY), (-4.75, LOBBY), (0, 165, 255)),  # bonus L
    ((4.75, -LOBBY), (4.75, LOBBY), (0, 165, 255)),    # bonus R
    ((-6.5, -5), (-6.5, 5), (0, 255, 0)),      # endline L
    ((6.5, -5), (6.5, 5), (0, 255, 0)),        # endline R
    ((-6.5, 5), (6.5, 5), (0, 255, 0)),        # sideline far
    ((-6.5, -5), (6.5, -5), (0, 255, 0)),      # sideline near
    ((-6.5, LOBBY), (6.5, LOBBY), (255, 0, 255)),      # lobby far
    ((-6.5, -LOBBY), (6.5, -LOBBY), (255, 0, 255)),    # lobby near
]

os.makedirs("data/overlays", exist_ok=True)
files = glob.glob("data/annotations/**/_annotations.coco.json", recursive=True)

for jf in files:
    folder = os.path.dirname(jf)
    data = json.load(open(jf))
    cats = {c["id"]: c for c in data["categories"]}
    images = {im["id"]: im["file_name"] for im in data["images"]}
    for ann in data["annotations"]:
        cat = cats[ann["category_id"]]
        if "keypoints" not in cat or "keypoints" not in ann:
            continue
        kp = ann["keypoints"]
        img_pts, court_pts = [], []
        for i, name in enumerate(cat["keypoints"]):
            x, y, v = kp[3*i], kp[3*i+1], kp[3*i+2]
            if v > 0 and not (x == 0 and y == 0) and name in COURT:
                img_pts.append([x, y]); court_pts.append(COURT[name])
        if len(img_pts) < 4:
            continue
        # court -> image homography
        Hc2i, _ = cv2.findHomography(np.array(court_pts, np.float32),
                                     np.array(img_pts, np.float32), 0)
        fn = images[ann["image_id"]]
        img = cv2.imread(os.path.join(folder, fn))
        if img is None:
            print("image open aagala:", fn); continue
        for (a, b, col) in LINES:
            pts = np.array([[a], [b]], np.float32)
            p = cv2.perspectiveTransform(pts, Hc2i).reshape(-1, 2)
            cv2.line(img, tuple(int(v) for v in p[0]), tuple(int(v) for v in p[1]), col, 1, cv2.LINE_AA)
        for x, y in img_pts:   # un click panna points (red)
            cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
        out = f"data/overlays/{fn[:22]}.jpg"
        cv2.imwrite(out, img)
        print("saved", out)