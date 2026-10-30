# Cắt nhân vật từ tranh gốc thành các tấm có khớp (puppet 2.5D): 3 đầu (kèm mũ/tai), thân+quần, 6 tay (kèm bàn tay/đạo cụ),
# 2 chân+giày, 2 đuôi+bùa, ngôi sao. Mỗi tấm: viền co 1px + làm mịn ⇒ đúng nét vẽ, ¬ răng cưa.
# Chạy: python3 tools/cut_parts.py   (Pillow + numpy). Ra: assets/parts/*.png, assets/parts.json
import json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REF = '../../jester3d/reference.jpg'
im = Image.open(REF).convert('RGB')
W, H = im.size
yy, xx = np.mgrid[0:H, 0:W]
ell = lambda cx, cy, rx, ry: ((xx-cx)/rx)**2 + ((yy-cy)/ry)**2 < 1

# ---- tiền cảnh: nền flood-fill từ góc + vùng kẹp giữa chân; bỏ bóng kem dưới giày ----
mask = im.copy()
for s in [(0,0),(W-1,0),(0,H-1),(W-1,H-1),(W//2,0),(W//2,H-1),(0,H//2),(W-1,H//2),(627,900),(627,1000)]:
    ImageDraw.floodfill(mask, s, (255,0,255), thresh=28)
a = np.asarray(im).astype(int); m = np.asarray(mask)
bg = (m[...,0]==255)&(m[...,1]==0)&(m[...,2]==255)
shadow = (yy>1040)&(a.min(2)>140)&((a.max(2)-a.min(2))<55)
fg = ~(bg|shadow)
left = fg.copy()   # phần chưa gán
# alpha chung: chỉ viền NGOÀI của nhân vật được co 1px + làm mịn; cạnh nối giữa các phần giữ đặc
ALPHA = Image.fromarray((fg*255).astype('uint8')).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.9))

def components(mk):
    # gán nhãn thành phần liền (4-láng giềng), không cần scipy
    from collections import deque
    lab = np.zeros(mk.shape, np.int32); n = 0
    ys, xs = np.nonzero(mk)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]: continue
        n += 1; lab[y0, x0] = n; q = deque([(y0, x0)])
        while q:
            y, x = q.popleft()
            for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                v, u = y+dy, x+dx
                if 0 <= v < mk.shape[0] and 0 <= u < mk.shape[1] and mk[v, u] and not lab[v, u]:
                    lab[v, u] = n; q.append((v, u))
    return lab, n

parts = {}
MASKS = {}
FUR = (22, 22, 28)  # lông đen của nhân vật
def emit(pid, mk, pivot, z, group, underlay=None):
    global left
    mk = mk & left
    left &= ~mk   # gán xong ⇒ bỏ khỏi vùng chưa gán, kể cả mảnh vụn (¬ rơi về thân đứng yên)
    # bỏ mảnh vụn rời < 120px (sợi viền sót khi chia vùng); chuông/ngón tay đều > 300px
    lab, n = components(mk)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        mk = mk & (sizes[lab] >= 120)
    MASKS[pid] = (mk, pivot)
    mi = Image.fromarray((mk*255).astype('uint8'))
    # nới 2px sang phần bên cạnh ⇒ các tấm chồng khít, ¬ hở khe ở đường cắt
    grow = np.asarray(mi.filter(ImageFilter.MaxFilter(5))) > 0
    alv = np.where(grow & fg, np.asarray(ALPHA), 0).astype('uint8')
    rgba = im.convert('RGBA')
    if underlay is not None:
        # lót lông đen dưới khớp vai/cổ ⇒ khi tay/đầu xoay, khe hở là lông, ¬ lộ nền
        u = np.asarray(Image.fromarray((underlay*255).astype('uint8')).filter(ImageFilter.GaussianBlur(1.2)))
        px = np.asarray(rgba).copy()
        hard = (underlay & (alv < 250))
        px[hard, :3] = FUR
        alv = np.maximum(alv, u)
        rgba = Image.fromarray(px, 'RGBA')
    al = Image.fromarray(alv)
    bbox = al.getbbox()
    rgba.putalpha(al)
    rgba.crop(bbox).save(f'assets/parts/{pid}.png')
    parts[pid] = {'file': f'assets/parts/{pid}.png', 'bbox': list(bbox), 'pivot': list(pivot), 'z': z, 'group': group}

os.makedirs('assets/parts', exist_ok=True)

# ---- ngôi sao bay ----
emit('star', fg & ell(1160,385,48,48), (1160,385), 0.22, 'prop')

# ---- tay: chia theo xương gần nhất; đạo cụ thuộc trọn tay dưới ----
heads_zone = ell(345,300,165,140) | ell(905,300,165,140)
skirt = (yy>690) & (xx>298) & (xx<960)
BONES = {
  'up-R':  ((470,480),(160,400)), 'mid-R': ((470,530),(320,545)), 'low-R': ((470,600),(130,680)),
  'up-L':  ((770,470),(1100,450)), 'mid-L': ((770,530),(890,580)), 'low-L': ((770,600),(1100,650)),
}
def segdist(p, q):
    (px, py), (qx, qy) = p, q; dx, dy = qx-px, qy-py
    t = np.clip(((xx-px)*dx + (yy-py)*dy)/(dx*dx+dy*dy), 0, 1)
    return np.hypot(xx-(px+t*dx), yy-(py+t*dy))
for s, side in (('R', xx<470), ('L', xx>770)):
    region = left & side & (yy>=300) & (yy<=760) & ~heads_zone & ~skirt
    ids = [k for k in BONES if k.endswith(s)]
    owner = np.stack([segdist(*BONES[k]) for k in ids]).argmin(0)
    prop = ell(128,553,74,74) if s == 'R' else ((xx>1045)&(xx<1222)&(yy>512)&(yy<698))
    owner[prop] = ids.index(f'low-{s}')
    # vòng cổ tay + chuông thuộc trọn tay của nó (¬ chia đôi theo xương)
    CUFF = {'R': {'up': (215,435,58,75), 'low': (262,680,48,60)}, 'L': {'up': (1025,480,60,75), 'low': (1020,690,52,55)}}
    for tier, (cx, cy, rx, ry) in CUFF[s].items():
        owner[ell(cx,cy,rx,ry)] = ids.index(f'{tier}-{s}')
    # mảnh rời (chuông, vòng, ngón) ⇒ giao trọn cho tay chiếm đa số; khối lớn nhất giữ phân chia theo xương
    lab, n = components(region)
    sizes = np.bincount(lab.ravel())[1:]
    big = 1 + int(sizes.argmax())
    for c in range(1, n+1):
        if c == big: continue
        cm = lab == c
        owner[cm] = np.bincount(owner[cm], minlength=len(ids)).argmax()
    for i, k in enumerate(ids):
        emit(k, region & (owner==i), BONES[k][0], 0.04 if k.startswith('mid') else 0.07, 'arm')

# ---- đầu: vùng trên cổ; chia theo tâm gần nhất (đầu giữa to hơn) ----
hz = left & ((yy<378) | ((xx<480)&(yy<430)) | ((xx<415)&(yy<445)) | ((xx>780)&(yy<410)) | ((xx>830)&(yy<432)))
C = {'head-w': ((335,305),1.0,(430,410),0.12), 'head-c': ((627,200),1.2,(627,378),0.17), 'head-t': ((925,305),1.0,(820,400),0.12)}
d = np.stack([np.hypot(xx-c[0][0], yy-c[0][1])/c[1] for c in C.values()])
own = d.argmin(0)
for i, (k, c) in enumerate(C.items()):
    emit(k, hz & (own==i), c[2], c[3], 'head')

# ---- đuôi + bùa ----
emit('tail-R', left & (xx<300) & (yy>740) & (yy<1005), (300,790), -0.12, 'tail')   # nửa trái ảnh = phải nhân vật
emit('tail-L', left & (xx>890) & (yy>740) & (yy<990), (890,800), -0.12, 'tail')

# ---- chân + giày ----
emit('leg-R', left & (yy>885) & (xx<640), (510,890), -0.04, 'leg')
emit('leg-L', left & (yy>885) & (xx>=640), (800,890), -0.04, 'leg')

# ---- thân: phần còn lại (cổ áo, nơ, dây đeo, quần, chuông váy) ----
# lớp lót dạng khớp tròn (¬ theo hình tay) ⇒ khi xoay chỉ lộ khối vai/cổ bo tròn, ¬ "cuống tay ma"
under = ell(472,545,46,105) | ell(768,540,46,105)          # sườn trái/phải
for k in BONES: under |= ell(*BONES[k][0], 52, 52)          # khớp vai
for k, c in C.items(): under |= ell(c[2][0], c[2][1]-10, 62, 40)  # cổ
emit('body', left.copy(), (627,1000), 0.0, 'body', underlay=under)

json.dump({'size':[W,H], 'pxPerUnit':300, 'origin':[627,1130], 'parts':parts}, open('assets/parts.json','w'), indent=1)
# ảnh kiểm tra: tô màu theo nhóm
O = '/tmp/claude-0/-home-user-content-ai/884b481b-e760-5bee-b880-c371a009e448/scratchpad/'
cols = {'head':(255,90,90),'arm':(90,170,255),'leg':(90,255,120),'tail':(255,220,60),'body':(200,120,255),'prop':(255,255,255)}
chk = Image.new('RGB',(W,H),(30,15,50))
for k,p in parts.items():
    t = Image.open(p['file']); tint = Image.new('RGB', t.size, cols[p['group']])
    chk.paste(Image.blend(t.convert('RGB'), tint, 0.4), tuple(p['bbox'][:2]), t)
chk.resize((627,627)).save(O+'parts-check.png')
print({k: p['bbox'] for k,p in parts.items()})
