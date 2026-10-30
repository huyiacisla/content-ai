# Tách Kai/Mốc (video2d cutout) thành khúc: head/torso/armL/armR/(legL/legR) + parts.json
# Chạy: python3 tools/puppet.py   (cần Pillow). Tọa độ px của ảnh gốc; pivot = khớp xoay.
import json
from PIL import Image, ImageDraw

SRC = 'assets/'
OUT = 'assets/puppet/'
# polygon theo ảnh gốc; viền cắt đặt dọc nếp áo/cổ ⇒ ít lộ
SPEC = {
  'kai': {
    'file': 'kai-cut.png', 'height': 4.6, 'fadeBottom': 0.1,
    'parts': {
      'head':  {'poly': [(80,0),(568,0),(568,540),(310,575),(250,575),(215,505),(160,440),(100,420),(80,300)], 'pivot': (285, 560)},
      'armL':  {'poly': [(60,880),(120,800),(170,820),(160,1000),(150,1100),(145,1290),(140,1380),(40,1380),(0,1290),(0,1130)], 'pivot': (140, 790)},
      'armR':  {'poly': [(420,760),(470,780),(500,880),(550,1000),(568,1120),(560,1250),(540,1360),(450,1360),(440,1290),(430,1100),(420,900)], 'pivot': (430, 775)},
    },
  },
  'moc': {
    'file': 'moc-cut.png', 'height': 4.3, 'fadeBottom': 0,
    'parts': {
      'head': {'poly': [(0,0),(609,0),(609,480),(300,500),(190,500),(0,480)], 'pivot': (245, 495)},
      'armL': {'poly': [(150,560),(100,640),(60,760),(70,850),(110,880),(160,860),(170,700)], 'pivot': (150, 575)},
      'armR': {'poly': [(300,570),(330,560),(380,640),(405,760),(395,850),(340,860),(320,800)], 'pivot': (320, 580)},
      'legL': {'poly': [(0,1045),(248,1045),(248,1298),(0,1298)], 'pivot': (170, 1045)},
      'legR': {'poly': [(248,1045),(609,1045),(609,1298),(248,1298)], 'pivot': (345, 1045)},
    },
  },
}

meta = {}
for who, sp in SPEC.items():
    im = Image.open(SRC + sp['file']).convert('RGBA')
    W, H = im.size
    taken = Image.new('L', (W, H), 0)
    meta[who] = {'size': [W, H], 'height': sp['height'], 'fadeBottom': sp['fadeBottom'], 'parts': {}}
    order = list(sp['parts'].items()) + [('torso', None)]
    for name, p in order:
        if p:
            m = Image.new('L', (W, H), 0)
            ImageDraw.Draw(m).polygon(p['poly'], fill=255)
            taken = Image.composite(Image.new('L', (W, H), 255), taken, m)
        else:  # torso = phần còn lại (kể cả chân nếu không tách)
            m = Image.eval(taken, lambda v: 255 - v)
        part = im.copy()
        a = Image.composite(part.getchannel('A'), Image.new('L', (W, H), 0), m)
        part.putalpha(a)
        bb = a.getbbox()
        part.crop(bb).save(f'{OUT}{who}_{name}.png')
        piv = p['pivot'] if p else (W / 2, H / 2)
        meta[who]['parts'][name] = {'file': f'./assets/puppet/{who}_{name}.png', 'bbox': list(bb), 'pivot': list(piv)}
json.dump(meta, open(OUT + 'parts.json', 'w'), indent=1)
print({w: list(m['parts']) for w, m in meta.items()})
