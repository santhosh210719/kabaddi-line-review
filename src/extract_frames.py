import cv2, os

def extract(video_path, out_dir, every_n_sec=0.5):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(fps * every_n_sec))
    name = os.path.splitext(os.path.basename(video_path))[0]
    i = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            cv2.imwrite(f"{out_dir}/{name}_{i:05d}.jpg", frame)
            saved += 1
        i += 1
    cap.release()
    print(f"{name}: {saved} frames")

if __name__ == "__main__":
    for f in sorted(os.listdir("data/raw_clips")):
        if f.lower().endswith((".mp4", ".mov", ".avi")):
            extract(f"data/raw_clips/{f}", "data/frames")