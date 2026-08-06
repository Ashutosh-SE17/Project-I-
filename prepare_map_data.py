"""
prepare_map_data.py
-------------------
One-time: shrink Nep_district.geojson so the browser can load it.

    python prepare_map_data.py

The source file is ~15.9 MB — 362,460 vertices stored at ~15 decimal
places, which is nanometre precision for a map viewed at zoom 7. Douglas-
Peucker simplification plus coordinate rounding brings it to ~0.3 MB with
no visible difference at national zoom.

Writes: static/data/nepal_districts.geojson
"""

import json
from pathlib import Path

from shapely.geometry import mapping, shape

BASE = Path(__file__).resolve().parent
SRC = BASE / 'Nep_district.geojson'
OUT = BASE / 'static' / 'data' / 'nepal_districts.geojson'

TOLERANCE = 0.002   # degrees; ~200 m — invisible at zoom 7
PRECISION = 5       # decimal places; ~1 m


def round_coords(obj, n=PRECISION):
    if isinstance(obj, float):
        return round(obj, n)
    if isinstance(obj, list):
        return [round_coords(x, n) for x in obj]
    return obj


def main():
    if not SRC.exists():
        raise SystemExit(f'Not found: {SRC}')

    data = json.loads(SRC.read_text(encoding='utf-8'))

    for feat in data['features']:
        geom = shape(feat['geometry']).simplify(TOLERANCE, preserve_topology=True)
        feat['geometry'] = json.loads(json.dumps(mapping(geom)))
        feat['geometry']['coordinates'] = round_coords(feat['geometry']['coordinates'])
        # keep only what the map actually reads
        props = feat['properties']
        feat['properties'] = {
            'district': props.get('DISTRICT'),
            'province': props.get('PR_NAME'),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')

    before = SRC.stat().st_size / 1e6
    after = OUT.stat().st_size / 1e6
    print(f'{before:.1f} MB -> {after:.2f} MB  ({100 * (1 - after / before):.0f}% smaller)')
    print(f'Saved -> {OUT}')


if __name__ == '__main__':
    main()
