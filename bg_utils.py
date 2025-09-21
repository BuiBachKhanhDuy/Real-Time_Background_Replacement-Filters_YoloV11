import numpy as np
import cv2

def apply_bg_to_masks(img, masks, bg_img):
    result = np.zeros_like(img)
    mask_combined = np.zeros(img.shape[:2], dtype=np.uint8)

    for mask in masks:
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask_bin = (mask_resized > 0.5).astype(np.uint8)
        mask_combined = cv2.bitwise_or(mask_combined, mask_bin)

    mask_3c = cv2.merge([mask_combined] * 3)
    bg_mask = cv2.bitwise_and(bg_img, bg_img, mask=(1 - mask_combined).astype(np.uint8))
    fg_mask = cv2.bitwise_and(img, img, mask=mask_combined)

    return cv2.add(bg_mask, fg_mask)
