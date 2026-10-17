import json, glob
import cv2
import numpy as np

LOBBY = 4.0   # lobby line y position (m). 4.0 or 4.25 try pannu

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

def is_collinear(pts):
    pts = np.asarray(pts, dtype=float)
    return np.linalg.matrix_rank(pts - pts.mean(axis=0), tol=1e-6) < 2

files = glob.glob("data/annotations/**/_annotations.coco.json", recursive=True)
if not files:
    print("JSON file kidaikkala, path check pannu")

for jf in files:
    data = json.load(open(jf))
    cats = {c["id"]: c for c in data["categories"]}
    images = {im["id"]: im["file_name"] for im in data["images"]}
    print("\n==", jf)
    print("categories:", [(c["id"], c["name"], "keypoints" in c) for c in data["categories"]])
    for ann in data["annotations"]:
        cat = cats[ann["category_id"]]
        if "keypoints" not in cat or "keypoints" not in ann:
            continue
        names = cat["keypoints"]
        kp = ann["keypoints"]
        img_pts, court_pts, used = [], [], []
        for i, name in enumerate(names):
            x, y, v = kp[3*i], kp[3*i+1], kp[3*i+2]
            if v > 0 and not (x == 0 and y == 0) and name in COURT:
                img_pts.append([x, y])
                court_pts.append(COURT[name])
                used.append(name)
        fn = images[ann["image_id"]]
        if len(img_pts) < 4 or is_collinear(court_pts):
            print(f"{fn[:40]:40s} SKIP ({len(img_pts)} points)")
            continue
        img_pts = np.array(img_pts, np.float32)
        court_pts = np.array(court_pts, np.float32)
        H, _ = cv2.findHomography(img_pts, court_pts, 0)
        proj = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = np.linalg.norm(proj - court_pts, axis=1)
        worst = used[int(err.argmax())]
        print(f"{fn[:40]:40s} pts={len(img_pts):2d} mean={err.mean()*100:5.1f} cm  worst={worst} ({err.max()*100:.1f} cm)")