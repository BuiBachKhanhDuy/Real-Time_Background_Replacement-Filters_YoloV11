import cv2
from ultralytics import YOLO
import os
import numpy as np
from bg_utils import apply_bg_to_masks
from yolo_utils import get_segmentation_masks
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from cvzone.FaceMeshModule import FaceMeshDetector  # for head detection

# Video capture setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load YOLOv11 segmentation model
model = YOLO(r"yolov11\yolo11n-seg.pt")

# Load background images
bg_list = [cv2.resize(cv2.imread(f'images/{img}'), (640, 480)) for img in os.listdir("images")]
index_bg = 0

# Load filter image (could be JPG or PNG)
orig_filter = cv2.imread('filters/cat.webp', cv2.IMREAD_UNCHANGED)
# If no alpha channel, generate one by thresholding nearly-white background
if orig_filter.shape[2] == 3:
    b, g, r = cv2.split(orig_filter)
    mask = ((b > 240) & (g > 240) & (r > 240)).astype(np.uint8) * 255
    alpha = cv2.bitwise_not(mask)
    orig_filter = cv2.merge((b, g, r, alpha))

# Toggles
enabled_bg = True
enabled_filter = False
multi_person = True

# Initialize modules
segmentor = SelfiSegmentation()
face_mesh = FaceMeshDetector(maxFaces=1)

# Video writer for side-by-side output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1) 
    img_out = frame.copy()

    # Background replacement 
    if enabled_bg:
        masks = get_segmentation_masks(model, frame)
        if multi_person:
            img_out = apply_bg_to_masks(frame, masks, bg_list[index_bg]) if masks else bg_list[index_bg].copy()
        else:
            if masks:
                H, W = frame.shape[:2]
                center = np.array([W/2, H/2])
                best, bd = None, float('inf')
                for m in masks:
                    ys, xs = np.where(m > 0.5)
                    if xs.size:
                        c = np.array([xs.mean(), ys.mean()])
                        d = np.linalg.norm(c - center)
                        if d < bd:
                            bd, best = d, m
                img_out = apply_bg_to_masks(frame, [best], bg_list[index_bg]) if best is not None else bg_list[index_bg].copy()
            else:
                img_out = bg_list[index_bg].copy()

    # Head filter overlay (scaled ×2) 
    if enabled_filter:
        overlay = img_out.copy()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_mesh, faces = face_mesh.findFaceMesh(img_rgb, draw=False)
        if faces:
            pts = np.array(faces[0])
            top = pts[10]
            bottom = pts[152]
            left = pts[234]
            right = pts[454]

            # Base box coords
            x1 = int(left[0])
            x2 = int(right[0])
            y1 = int(top[1] - 0.25 * (bottom[1] - top[1]))
            y2 = int(bottom[1] + 0.05 * (bottom[1] - top[1]))

            # Compute width/height and scale ×2
            fw = x2 - x1
            fh = y2 - y1
            fw2 = int(fw * 2)
            fh2 = int(fh * 2)

            # Center the box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            nx1 = max(0, cx - fw2 // 2)
            ny1 = max(0, cy - fh2 // 2)
            nx2 = min(frame.shape[1], nx1 + fw2)
            ny2 = min(frame.shape[0], ny1 + fh2)

            # Adjust if box goes out of bounds
            fw2 = nx2 - nx1
            fh2 = ny2 - ny1

            if fw2 > 0 and fh2 > 0:
                filt = cv2.resize(orig_filter, (fw2, fh2), interpolation=cv2.INTER_AREA)
                b, g, r, a = cv2.split(filt)
                fg = cv2.merge((b, g, r))
                mask = a.astype(float) / 255.0
                for c in range(3):
                    overlay[ny1:ny2, nx1:nx2, c] = (
                        fg[:, :, c] * mask + overlay[ny1:ny2, nx1:nx2, c] * (1 - mask)
                    )
                img_out = overlay

    # Display
    stacked = np.hstack((frame, img_out))
    label = f"{'Multi' if multi_person else 'Single'} | BG:{'ON' if enabled_bg else 'OFF'} | Filter:{'ON' if enabled_filter else 'OFF'}"
    cv2.putText(stacked, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('BG & Filter', stacked)
    out.write(stacked)

    key = cv2.waitKey(1)
    if key == ord('1'):
        index_bg = (index_bg - 1) % len(bg_list)
    elif key == ord('2'):
        index_bg = (index_bg + 1) % len(bg_list)
    elif key == ord('m'):
        multi_person = not multi_person
    elif key == ord('b'):
        enabled_bg = not enabled_bg
    elif key == ord('f'):
        enabled_filter = not enabled_filter
    elif key == ord('x'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
