"""Portable CPU ONNX reference for this bundle. No test-prj imports or PyTorch."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

if __package__:
    from .ops import preprocess, decode, preprocess_meter, ordered_slot_crops, decode_logits
    from .visionstate_common import cell_input
else:
    from ops import preprocess, decode, preprocess_meter, ordered_slot_crops, decode_logits
    from visionstate_common import cell_input


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def format_reading(raw, decimal_places):
    if len(raw) <= decimal_places:
        return None
    return raw[:-decimal_places] + '.' + raw[-decimal_places:] if decimal_places else raw


class MeterPipeline:
    """Load once and reuse. predict() accepts PIL images; predict_file() decodes files."""
    def __init__(self, models_dir=None, threads=4):
        if threads < 1:
            raise ValueError('threads must be positive')
        self.root = Path(models_dir) if models_dir else Path(__file__).resolve().parents[1]
        self.manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        self.config = self.manifest['pipeline']
        self.sessions = {}
        for role, record in self.manifest['models'].items():
            path = self.root / record['file']
            if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
                raise ValueError(f'Model size/hash mismatch: {path}')
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1 if role == 'wheel_reader' else threads
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=['CPUExecutionProvider'])
            for kind, actual in [('inputs', session.get_inputs()), ('outputs', session.get_outputs())]:
                expected = record[kind]
                if len(actual) != len(expected):
                    raise ValueError(f'{role}: wrong {kind} count')
                for tensor, spec in zip(actual, expected):
                    if (tensor.name != spec['name'] or tensor.type != 'tensor(float)'
                            or tensor.shape != spec['shape']):
                        raise ValueError(f'{role}: unexpected tensor contract: {tensor.name}')
            self.sessions[role] = session

    def _run(self, role, tensor):
        session = self.sessions[role]
        return session.run(None, {session.get_inputs()[0].name: tensor})[0]

    def _detect(self, role, rgb, names, agnostic=True):
        tensor, scale, left, top = preprocess(rgb[:, :, ::-1].copy(), 640, 640)
        output = self._run(role, tensor)
        return decode(output, scale, left, top, rgb.shape, names,
                      self.config['confidence_threshold'], self.config['nms_iou_threshold'], agnostic)

    def predict_file(self, path):
        path = Path(path)
        with Image.open(path) as source:
            result = self.predict(source)
        return dict(image=str(path.resolve()), source_sha256=sha256(path), **result)

    def predict(self, image):
        """Correct EXIF orientation here; see PIPELINE.md for each coordinate field's frame."""
        image = ImageOps.exif_transpose(image).convert('RGB')
        rgb = np.asarray(image)
        h, w = rgb.shape[:2]
        tensor, _ = preprocess_meter(image)
        logits = self._run('classifier', tensor)[0]
        scores = np.exp(logits - logits.max())
        scores /= scores.sum()
        types = ['digital', 'wheel']
        meter_type = types[int(scores.argmax())]
        candidates, best = self._detect('area', rgb, {0: 'area'})
        selected = max(candidates, key=lambda d: (d['score'],
            (d['xyxy'][2]-d['xyxy'][0])*(d['xyxy'][3]-d['xyxy'][1])), default=None)
        result = dict(width=w, height=h, meter_type=meter_type,
                      type_scores=dict(zip(types, map(float, scores))),
                      area_candidates=candidates, area=selected, area_best_raw_score=best,
                      raw_digits=None, reading=None, status='no_area', reasons=[],
                      digit_detections=[], slots=[])
        if selected is None:
            return result
        box = selected['xyxy']
        pad_x = pad_y = 0.
        if meter_type == 'wheel':
            context = self.config['wheel_area_context']['horizontal_fraction_per_side']
            pad_x = (box[2]-box[0])*context
            pad_y = (box[3]-box[1])*context*7/6
        a, b = max(0, math.floor(box[0]-pad_x)), max(0, math.floor(box[1]-pad_y))
        c, d = min(w, math.ceil(box[2]+pad_x)), min(h, math.ceil(box[3]+pad_y))
        area = rgb[b:d, a:c].copy()
        result['area_crop_xyxy'] = [a, b, c, d]
        if meter_type == 'digital':
            names = dict(enumerate([str(i) for i in range(10)] + ['dot', 'minus']))
            symbols, _ = self._detect('lcd_reader', area, names)
            digits = sorted((s for s in symbols if len(s['class_name']) == 1
                             and s['class_name'] in '0123456789'),
                            key=lambda s: (s['xyxy'][0]+s['xyxy'][2])/2)
            raw = ''.join(s['class_name'] for s in digits)
            result.update(raw_digits=raw, reading=format_reading(raw, 2),
                          all_symbols=symbols, digit_detections=digits)
            if a <= 0 or b <= 0 or c >= w or d >= h:
                result['reasons'].append('window_touches_image_edge')
            if len(raw) < 3:
                result['reasons'].append('too_few_digits')
            result['status'] = 'review' if result['reasons'] else 'candidate'
            return result
        detected, best = self._detect('wheel_slots', area, {0: 'wheel_slot'}, agnostic=False)
        local_slots = ordered_slot_crops(area, detected)
        global_boxes = [{'xyxy': (np.asarray(s['xyxy'])+[a, b, a, b]).tolist(),
                         'score': s['score']} for s in local_slots]
        margin = self.config['wheel_crop_margin']
        slots = ordered_slot_crops(rgb, global_boxes, margin['width_fraction_per_side'],
                                   margin['height_fraction_per_side'])
        result.update(slot_best_raw_score=best, status='review' if slots else 'no_slots')
        if not slots:
            return result
        cells = np.stack([cell_input(Image.fromarray(s['pixels'])) for s in slots])[:, None]
        prediction = decode_logits(self._run('wheel_reader', cells))
        result.update(prediction=prediction, raw_digits=prediction['raw_digits'],
                      reading=format_reading(prediction['raw_digits'], 1))
        if result['reading'] is None:
            result['reasons'].append('too_few_slots_for_decimal_format')
        if len(slots) not in (5, 6):
            result['reasons'].append('unusual_slot_count_review_required')
        for i, (local, full) in enumerate(zip(local_slots, slots)):
            result['slots'].append(dict(index=i, xyxy_in_area=local['xyxy'],
                xyxy_in_image=full['xyxy'], crop_xyxy_in_image=full['crop_xyxy'],
                score=full['score'], independent_state=prediction['independent_states'][i]))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, help='Bundle directory (default: parent of reference/)')
    parser.add_argument('--input', type=Path, nargs='+', help='Image files or folders (recursive)')
    parser.add_argument('--output', type=Path, help='New output JSON; existing files are protected')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--verify-only', action='store_true', help='Check hashes and session contracts')
    args = parser.parse_args()
    if not args.verify_only and (not args.input or not args.output):
        parser.error('--input and --output are required unless --verify-only')
    if args.output and args.output.exists():
        parser.error('Choose a new output file')
    pipeline = MeterPipeline(args.models, args.threads)
    if args.verify_only:
        print('OK: 5 model hashes and ONNX Runtime input/output contracts')
        return
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    paths = sorted({p.resolve() for item in args.input
        for p in (item.rglob('*') if item.is_dir() else [item])
        if p.is_file() and p.suffix.lower() in extensions})
    if not paths:
        parser.error('No supported images found')
    rows = []
    for path in paths:
        row = pipeline.predict_file(path)
        rows.append(row)
        print(path.name, row['meter_type'], row['status'], row['reading'], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(bundle_version=pipeline.manifest['bundle_version'],
        runtime=ort.__version__, images=rows), ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
