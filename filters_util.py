import cv2
import numpy as np

def load_png_with_alpha(path):
    """Load a PNG with alpha channel."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Filter asset not found: {path}")
    return img

def get_head_position(mask):
    """Find the centroid of the topmost region of a person mask."""
    ys, xs = np.where(mask > 0.5)
    if ys.size == 0:
        return None
    top_y = ys.min()
    xs_at_top = xs[ys == top_y]
    return int(xs_at_top.mean()), int(top_y)

def overlay_png(bg, png, x, y, scale=1.0):
    """
    Overlay a RGBA png onto bg at center (x,y), scaled by `scale`.
    """
    h_bg, w_bg = bg.shape[:2]
    h, w = int(png.shape[0] * scale), int(png.shape[1] * scale)
    png_resized = cv2.resize(png, (w, h), interpolation=cv2.INTER_AREA)

    # compute placement
    x0, y0 = int(x - w / 2), int(y - h / 2)
    x1, y1 = max(0, x0), max(0, y0)
    x2, y2 = min(w_bg, x0 + w), min(h_bg, y0 + h)

    roi_bg = bg[y1:y2, x1:x2]
    roi_png = png_resized[(y1 - y0):(y2 - y0), (x1 - x0):(x2 - x0)]

    b, g, r, a = cv2.split(roi_png)
    alpha = a.astype(float) / 255.0

    # blend channels
    for c, channel in enumerate((b, g, r)):
        roi_bg[:, :, c] = (alpha * channel + (1 - alpha) * roi_bg[:, :, c]).astype(np.uint8)

    bg[y1:y2, x1:x2] = roi_bg
    return bg
