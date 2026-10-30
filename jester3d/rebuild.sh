#!/usr/bin/env bash
# build_spec → PBR evidence → albedo phẳng → finish → strict validate → factory
set -e
SK=/root/.claude/skills/img2threejs
cd "$(dirname "$0")"
python3 build_spec.py
for m in fur-black face-cream cloth-teal cloth-orange cloth-purple gold-metal bone tongue-pink shoe-satin ink-white; do
  python3 $SK/forge/stage1_intake/extract_pbr_evidence.py crops/$m.png --out-dir material-evidence/$m --material-id $m --target-threshold 0.7 --spec object-sculpt-spec.json --in-place --report material-evidence/$m.json >/dev/null
done
python3 - <<'PY'
import json
from PIL import Image
s=json.load(open('object-sculpt-spec.json'))
for m in s['materials']:
    h=m['baseColor'].lstrip('#'); rgb=tuple(int(h[i:i+2],16) for i in (0,2,4))
    Image.new('RGB',(1024,1024),rgb).save(f"material-evidence/{m['id']}/{m['id']}_albedo.png")
    m['referencePbr']['albedoNote']='albedo map flattened to the dominant cel-shaded fill colour (source is flat toon colour; crop texture would inject neighbouring colours)'
json.dump(s,open('object-sculpt-spec.json','w'),indent=1,ensure_ascii=False)
PY
python3 finish_spec.py
python3 $SK/forge/stage2_spec/validate_sculpt_spec.py object-sculpt-spec.json --strict-quality | tail -1
python3 $SK/forge/stage3_build/generate_threejs_factory.py object-sculpt-spec.json --out src/createObjectModel.ts --pass-id blockout --force >/dev/null
echo factory-ok
