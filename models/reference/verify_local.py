"""Replay local images with packaged models; compare to the frozen project results."""
import argparse
import copy
import json
import platform
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import PIL
from PIL import Image, ImageOps

if __package__:
    from .pipeline import MeterPipeline, sha256
    from .visionstate_common import cell_input
else:
    from pipeline import MeterPipeline, sha256
    from visionstate_common import cell_input


def main():
    bundle = Path(__file__).resolve().parents[1]
    repo = bundle.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=repo/'test-prj/results/all_local_meters')
    parser.add_argument('--data', type=Path, default=repo/'datas')
    parser.add_argument('--output', type=Path, default=bundle/'verification.json')
    args = parser.parse_args()
    evaluation = json.loads((args.baseline/'evaluation.json').read_text(encoding='utf-8'))
    area_rows = json.loads((args.baseline/'area/detections.json').read_text(encoding='utf-8'))['images']
    area_rows = {r['source_sha256']: r for r in area_rows}
    digit_rows = json.loads((args.baseline/'digital/readings.json').read_text(encoding='utf-8'))['images']
    digit_rows = {Path(r['image']).name: r for r in digit_rows}
    wheel_rows = json.loads((args.baseline/'wheel/readings.json').read_text(encoding='utf-8'))['images']
    wheel_rows = {r['source_sha256']: r for r in wheel_rows}
    pipeline = MeterPipeline(bundle)
    mismatches, examples = [], {}
    counts = dict(images=0, type_matches=0, raw_digit_matches=0, reading_matches=0,
                  area_matches=0, lcd_digit_box_sets=0, wheel_crop_box_sets=0,
                  wheel_logits_sets=0, wheel_input_cells=0, scorable=0, correct=0)
    for expected in evaluation['images']:
        path = args.data/expected['relative_image']
        if sha256(path) != expected['source_sha256']:
            raise ValueError('Source image changed: '+str(path))
        actual = pipeline.predict_file(path)
        area = area_rows[expected['source_sha256']]
        checks = {
            'type': actual['meter_type'] == expected['predicted_type'],
            'raw_digits': actual['raw_digits'] == expected['raw_digits'],
            'reading': actual['reading'] == expected['candidate'],
            'area': bool(actual['area']) == expected['has_area'],
        }
        if checks['area'] and actual['area']:
            checks['area'] = bool(np.allclose(actual['area']['xyxy'], area['detections'][0]['xyxy'],
                                             rtol=0, atol=1e-4))
        if not np.allclose(list(actual['type_scores'].values()), list(area['type_scores'].values()),
                           rtol=0, atol=1e-6):
            checks['type_scores'] = False
        counts['images'] += 1
        for key, check in [('type_matches', 'type'), ('raw_digit_matches', 'raw_digits'),
                           ('reading_matches', 'reading'), ('area_matches', 'area')]:
            counts[key] += int(checks[check])
        if expected['reference_raw'] is not None:
            counts['scorable'] += 1
            counts['correct'] += int(actual['raw_digits'] == expected['reference_raw']
                                     and actual['meter_type'] == expected['reference_type'])
        if actual['area'] and actual['meter_type'] == 'digital':
            original = digit_rows[path.name]
            checks['lcd_boxes'] = len(actual['digit_detections']) == len(original['digit_detections'])
            if checks['lcd_boxes']:
                checks['lcd_boxes'] = all(a['class_id'] == e['class_id'] and
                    np.allclose(a['xyxy'], e['xyxy'], rtol=0, atol=1e-4)
                    for a, e in zip(actual['digit_detections'], original['digit_detections']))
            counts['lcd_digit_box_sets'] += int(checks['lcd_boxes'])
        if actual['meter_type'] == 'wheel' and actual.get('prediction'):
            original = wheel_rows[expected['source_sha256']]
            checks['wheel_crops'] = len(actual['slots']) == len(original['slots']) and all(
                a['crop_xyxy_in_image'] == e['crop_xyxy_in_image']
                for a, e in zip(actual['slots'], original['slots']))
            counts['wheel_crop_box_sets'] += int(checks['wheel_crops'])
            checks['wheel_logits'] = bool(np.allclose(actual['prediction']['logits'],
                original['prediction']['logits'], rtol=0, atol=1e-6))
            counts['wheel_logits_sets'] += int(checks['wheel_logits'])
            with Image.open(path) as image:
                rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
            for slot, old in zip(actual['slots'], original['slots']):
                a, b, c, d = slot['crop_xyxy_in_image']
                tensor = cell_input(Image.fromarray(rgb[b:d, a:c]))
                old_path = Path(old['model_input'])
                if not old_path.is_absolute():
                    old_path = repo/'test-prj'/old_path
                with Image.open(old_path) as image:
                    saved = np.asarray(image.convert('L'))
                same = bool(np.array_equal(np.rint(tensor*255).astype(np.uint8), saved))
                checks['wheel_input_'+str(slot['index'])] = same
                counts['wheel_input_cells'] += int(same)
        if not all(checks.values()):
            mismatches.append(dict(image=expected['relative_image'],
                failed=[k for k, value in checks.items() if not value]))
        kind = ('digital' if actual['meter_type'] == 'digital' else 'wheel')
        if actual['area'] is None:
            kind = 'no_area'
        elif expected['reference_raw'] == '003733' and actual['raw_digits'] == '003732':
            kind = 'wheel_known_floor_issue'
        if kind not in examples and (kind in ('no_area', 'wheel_known_floor_issue') or expected['exact_match']):
            example = copy.deepcopy(actual)
            example['image'] = expected['relative_image']
            if example.get('prediction'):
                for k in ('logits', 'probabilities'):
                    example['prediction'].pop(k)
            examples[kind] = dict(reference_raw=expected['reference_raw'],
                note='Illustrative replay; model output may be incorrect. Full logits omitted.', output=example)
        print(f"{counts['images']:3d} {path.name}: {'OK' if all(checks.values()) else checks}", flush=True)
    report = dict(status='passed' if not mismatches else 'failed',
        bundle_version=pipeline.manifest['bundle_version'],
        runtime=dict(python=platform.python_version(), onnxruntime=ort.__version__, numpy=np.__version__,
                     opencv=cv2.__version__, pillow=PIL.__version__, provider='CPUExecutionProvider'),
        model_hashes_and_session_contracts='passed (all five models)',
        model_sha256={role: item['sha256'] for role, item in pipeline.manifest['models'].items()},
        reference_file_sha256={p.name: sha256(p) for p in sorted((bundle/'reference').glob('*.py'))},
        counts=counts, mismatches=mismatches,
        tolerances=dict(type_scores_absolute=1e-6, box_coordinates_absolute_pixels=1e-4,
                        wheel_logits_absolute=1e-6, wheel_input_pixels='exact uint8 equality'),
        baseline_sha256={str(p.relative_to(args.baseline)).replace('\\', '/'): sha256(p) for p in
            [args.baseline/'evaluation.json', args.baseline/'area/detections.json',
             args.baseline/'digital/readings.json', args.baseline/'wheel/readings.json']},
        note='Packaging equivalence check, not independent accuracy measurement or phone validation.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    example_dir = args.output.parent/'examples'
    example_dir.mkdir(parents=True, exist_ok=True)
    for kind, value in examples.items():
        (example_dir/(kind+'.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
