import torch

def get_segmentation_masks(model, frame, allowed_classes={0, 15, 16}):
    results = model(frame)
    masks = []
    for r in results:
        if r.masks is not None:
            for i, mask in enumerate(r.masks.data):
                cls_id = int(r.boxes.cls[i])
                if cls_id in allowed_classes:
                    masks.append(mask.cpu().numpy())
    return masks

