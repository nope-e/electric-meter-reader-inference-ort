"""Frozen preprocessing/postprocessing from the tested project implementation.
See manifest.json reference_sources for original file SHA256.
"""
import math
import cv2
import numpy as np
from PIL import Image,ImageOps
if __package__:
    from .visionstate_common import cell_input,decode_shifted
else:
    from visionstate_common import cell_input,decode_shifted

def preprocess(image, height, width):
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized_w, resized_h = round(w * scale), round(h * scale)
    left = round((width - resized_w) / 2 - 0.1)
    top = round((height - resized_h) / 2 - 0.1)
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_h, left:left + resized_w] = cv2.resize(
        image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
    )
    tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= 255.0
    # Rounding resize dimensions gives slightly different effective X/Y scales.
    return tensor, (resized_w / w, resized_h / h), left, top


def decode(output, scale, left, top, image_shape, class_names, confidence, iou, agnostic=False):
    # Standard Ultralytics detection export, nms=False: [1, 4 + classes, anchors].
    expected = 4 + len(class_names)
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != expected:
        raise ValueError(f"Expected [1, {expected}, anchors], got {output.shape}")
    rows = output[0].T
    scores = rows[:, 4:].max(axis=1)
    best_raw_score = float(scores.max())
    rows = rows[scores >= confidence]
    if len(rows) == 0:
        return [], best_raw_score
    scores = rows[:, 4:].max(axis=1)
    classes = rows[:, 4:].argmax(axis=1)
    xywh = rows[:, :4].copy()
    xywh[:, :2] -= xywh[:, 2:] / 2
    kept = []
    groups = [np.arange(len(rows))] if agnostic else [np.flatnonzero(classes == c) for c in np.unique(classes)]
    for indices in groups:
        selected = cv2.dnn.NMSBoxes(
            xywh[indices].tolist(), scores[indices].tolist(), confidence, iou
        )
        kept.extend(indices[np.asarray(selected, dtype=int).reshape(-1)].tolist())
    h, w = image_shape[:2]
    scale_x, scale_y = scale
    detections = []
    for index in sorted(kept, key=lambda i: float(scores[i]), reverse=True):
        x, y, bw, bh = xywh[index]
        box = [
            max(0.0, min(float(w), float((x - left) / scale_x))),
            max(0.0, min(float(h), float((y - top) / scale_y))),
            max(0.0, min(float(w), float((x + bw - left) / scale_x))),
            max(0.0, min(float(h), float((y + bh - top) / scale_y))),
        ]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        class_id = int(classes[index])
        detections.append({
            "xyxy": box, "score": float(scores[index]),
            "class_id": class_id, "class_name": class_names[class_id],
        })
    return detections, best_raw_score


def preprocess_meter(image):
    """PIL RGB full image -> fixed NCHW tensor; identical to training."""
    image = ImageOps.pad(image.convert('RGB'), (256, 256),
                         method=Image.Resampling.BILINEAR, color=(128, 128, 128))
    values = np.asarray(image, dtype=np.float32) / np.float32(255)
    values = (values - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None]), image


def ordered_slot_crops(rgb,detections,margin_x=0.,margin_y=0.):
    """Use every nonempty post-NMS box, left to right; never extrapolate a wheel."""
    h,w=rgb.shape[:2]
    if margin_x<0 or margin_y<0: raise ValueError('Negative slot margin')
    slots=[]
    for detection in sorted(detections,key=lambda d:((d['xyxy'][0]+d['xyxy'][2])/2,(d['xyxy'][1]+d['xyxy'][3])/2)):
        box=np.asarray(detection['xyxy'],float)
        if box.shape!=(4,) or not np.isfinite(box).all(): raise ValueError('Invalid slot box')
        padding=(box[2:]-box[:2])*[margin_x,margin_y]
        a,b=np.maximum([0,0],np.floor(box[:2]-padding).astype(int))
        c,d=np.minimum([w,h],np.ceil(box[2:]+padding).astype(int))
        if c<=a or d<=b: continue
        slots.append({'xyxy':box.tolist(),'crop_xyxy':[int(a),int(b),int(c),int(d)],
            'score':float(detection['score']),'pixels':rgb[b:d,a:c].copy()})
    return slots


def decode_logits(logits):
    logits = np.asarray(logits, np.float64)
    if logits.ndim!=2 or logits.shape[1]!=100 or not len(logits) or not np.isfinite(logits).all():
        raise ValueError('Expected nonempty wheels x 100 finite logits')
    exponential = np.exp(logits-logits.max(axis=1,keepdims=True))
    probabilities = exponential/exponential.sum(axis=1,keepdims=True)
    # Author's release defaults from settings.py at the pinned code revision.
    near = sum(np.roll(probabilities,s,axis=1) for s in (-1,0,1))
    logp = np.log(np.maximum(near,1e-12))
    value,score,shift = decode_shifted(logp,max_shift=2,penalty=.3)
    digits = len(logits)
    raw_bins = np.argmax(logits,axis=1)
    return {'independent_raw_digits':''.join(str(int(b)//10) for b in raw_bins),
            'independent_states':(raw_bins/10).tolist(),
            'joint_value_in_last_wheel_units':value,'joint_log_score':score,
            'common_shift_bins':shift,
            'raw_digits':f'{math.floor(value)%10**digits:0{digits}d}',
            'author_rounded_raw_digits':f'{math.floor(value+.5)%10**digits:0{digits}d}',
            'logits':logits.tolist(),'probabilities':probabilities.tolist()}
