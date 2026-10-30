# Cắt 6 tay (kèm bàn tay/vòng/bóng/bài) + ngôi sao từ ảnh gốc thành tấm alpha riêng; xóa chúng khỏi texture chiếu thân.
# Chạy: python3 tools/arm_cards.py   (cần Pillow + numpy). Ra: assets/jester-proj.png, assets/arms/*.png, assets/arms.json
import json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REF = '../../jester3d/reference.jpg'
im = Image.open(REF).convert('RGB')
W, H = im.size

# nền: flood-fill từ góc + vùng kẹp giữa chân; bóng kem dưới giày
mask = im.copy()
for s in [(0,0),(W-1,0),(0,H-1),(W-1,H-1),(W//2,0),(W//2,H-1),(0,H//2),(W-1,H//2),(627,900),(627,1000)]:
    ImageDraw.floodfill(mask, s, (255,0,255), thresh=28)
a = np.asarray(im).astype(int); m = np.asarray(mask)
bg = (m[...,0]==255)&(m[...,1]==0)&(m[...,2]==255)
yy, xx = np.mgrid[0:H, 0:W]
shadow = (yy>1040)&(a.min(2)>150)&((a.max(2)-a.min(2))<45)
fg = ~(bg|shadow)

# vùng tay: ngoài mép thân (trái x<470, phải x>770), y 300..760, trừ đầu, trừ váy/quần (y>705 chỉ giữ chuông cổ tay)
def ell(cx, cy, rx, ry): return ((xx-cx)/rx)**2 + ((yy-cy)/ry)**2 < 1
heads = ell(345,300,165,140) | ell(905,300,165,140)
side = {'R': (xx<470), 'L': (xx>770)}   # trái/phải theo nhân vật: nửa ảnh trái = tay PHẢI nhân vật
skirt = (yy>675) & (xx>270) & (xx<990)
# xương tay (px ảnh): vai → bàn tay; dùng để chia pixel về 3 tay theo khoảng cách gần nhất
BONES = {
  'up-R':  ((470,480),(160,400)), 'mid-R': ((470,530),(320,545)), 'low-R': ((470,600),(130,680)),
  'up-L':  ((770,470),(1100,450)), 'mid-L': ((770,530),(890,580)), 'low-L': ((770,600),(1100,650)),
}
def segdist(p, q):
    px, py = p; qx, qy = q; dx, dy = qx-px, qy-py
    t = np.clip(((xx-px)*dx + (yy-py)*dy)/(dx*dx+dy*dy), 0, 1)
    return np.hypot(xx-(px+t*dx), yy-(py+t*dy))
out = {'size':[W,H], 'arms':{}}
proj_alpha = fg.copy()
import os; os.makedirs('assets/arms', exist_ok=True)
for s in ['R','L']:
    region = fg & side[s] & (yy>=300) & (yy<=760) & ~heads & ~skirt
    if s == 'L': region &= ~ell(1160,385,45,45)   # ngôi sao bay riêng
    ids = [k for k in BONES if k.endswith(s)]
    d = np.stack([segdist(*BONES[k]) for k in ids])
    owner = d.argmin(0)
    # đạo cụ thuộc trọn tay dưới (¬ chia theo khoảng cách): bóng sao bên phải nhân vật, xấp bài bên trái
    prop = ell(128,553,74,74) if s == 'R' else ((xx>1045)&(xx<1222)&(yy>512)&(yy<698))
    owner[prop] = ids.index(f'low-{s}')
    for i, k in enumerate(ids):
        mk = region & (owner==i)
        # giữ thành phần liền lớn nhất + mọi mảnh chạm nó (bóng/bài/chuông) — đơn giản: bỏ mảnh < 60 px
        mki = Image.fromarray((mk*255).astype('uint8'))
        al = mki.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.9))  # co 1px + làm mịn viền
        bbox = mki.getbbox()
        rgba = im.convert('RGBA'); rgba.putalpha(al)
        rgba.crop(bbox).save(f'assets/arms/{k}.png')
        (sx, sy), _ = BONES[k]
        out['arms'][k] = {'file': f'assets/arms/{k}.png', 'bbox': list(bbox), 'pivot': [sx, sy]}
        proj_alpha &= ~mk
# ngôi sao
st = fg & ell(1160,385,48,48)
sti = Image.fromarray((st*255).astype('uint8'))
al = sti.filter(ImageFilter.GaussianBlur(0.9)); bb = sti.getbbox()
rg = im.convert('RGBA'); rg.putalpha(al); rg.crop(bb).save('assets/arms/star.png')
out['star'] = {'file':'assets/arms/star.png','bbox':list(bb)}
proj_alpha &= ~st
# texture thân: co 1px + làm mịn viền ⇒ hết răng cưa/viền sáng
pa = Image.fromarray((proj_alpha*255).astype('uint8')).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.9))
body = im.convert('RGBA'); body.putalpha(pa); body.save('assets/jester-proj.png')
json.dump(out, open('assets/arms.json','w'), indent=1)
# ảnh kiểm tra: tô mỗi tay 1 màu
chk = Image.new('RGB',(W,H),(40,20,70)); chk.paste(im, (0,0), pa)
cols = {'up':(255,80,80),'mid':(80,255,80),'low':(80,160,255)}
for k,v in out['arms'].items():
    t = Image.open(v['file']); tint = Image.new('RGB', t.size, cols[k.split('-')[0]])
    chk.paste(Image.blend(t.convert('RGB'), tint, 0.45), tuple(v['bbox'][:2]), t)
chk.resize((627,627)).save('/tmp/claude-0/-home-user-content-ai/884b481b-e760-5bee-b880-c371a009e448/scratchpad/arms-check.png')
print({k: v['bbox'] for k, v in out['arms'].items()})
