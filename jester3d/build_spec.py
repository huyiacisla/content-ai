# Dựng object-sculpt-spec.json từ starter-spec.json (tọa độ: px ảnh → đơn vị: x=(px-627)/300, y=(1130-py)/300, z hướng ra trước)
import json, math, copy
import numpy as np
WORLD = {}   # id -> (pos np.array(3), R np.array(3x3), scale np.array(3))
def eul(r):
    cx,cy,cz=[math.cos(v) for v in r]; sx,sy,sz=[math.sin(v) for v in r]
    Rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]]); Ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]); Rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rx@Ry@Rz

s = json.load(open('starter-spec.json'))
tmpl = copy.deepcopy(s['componentTree'][0])
U = lambda px, py: [round((px - 627) / 300, 3), round((1130 - py) / 300, 3)]
PAL = dict(black='#16161c', cream='#efe6d0', teal='#1f8a86', orange='#f08a1c', purple='#6a3fa0', gold='#d9a520', bone='#e8dcc0', pink='#e58a9a', white='#ffffff', brown='#7a4a14')
mats = []
def mat(id, color, rough, metal=0.0, name=None):
    m = copy.deepcopy(s['materials'][0])
    m.update(id=id, name=name or id, baseColor=color, color=color, roughness=rough, metalness=metal)
    m['albedo'] = dict(m['albedo'], dominant=color, secondary=[color])
    m['colorVariation'] = dict(m['colorVariation'], palette=[color], amplitude=0.04, pattern='flat-toon')
    m['localOverrides'] = []
    mats.append(m)
mat('fur-black', PAL['black'], 0.85, name='Black cartoon fur'); mat('face-cream', PAL['cream'], 0.8)
mat('cloth-teal', PAL['teal'], 0.75); mat('cloth-orange', PAL['orange'], 0.75); mat('cloth-purple', PAL['purple'], 0.75)
mat('gold-metal', PAL['gold'], 0.3, 1.0); mat('bone', PAL['bone'], 0.6); mat('tongue-pink', PAL['pink'], 0.5)
mat('shoe-satin', PAL['teal'], 0.4, 0.05); mat('ink-white', PAL['white'], 0.4)
comps = []
def add(id, name, level, prim, parent, pos, dim, mat_id, rot=(0,0,0), scale=(1,1,1), role='part', extra=None, feats=None):
    c = copy.deepcopy(tmpl)
    c.update(id=id, name=name, level=level, role=role, primitive=prim, parent=parent, material=mat_id, materialLayers=[mat_id], fidelityTier='blockout', confidence=0.7)
    c['topologyClass'] = 'continuous-sculpt' if prim in ('ellipsoid','sphere','capsule') else 'assembled-solid'
    c['topologyRationale'] = 'Stylized toon part read from the reference silhouette (surface_topology: continuous organic form → ellipsoid/capsule; hard accessories → assembled solids).'
    if prim == 'capsule': dim = (dim[0]/0.7, dim[1]/1.4, dim[2]/0.7)   # CapsuleGeometry(0.35,0.7) = 0.7 rộng × 1.4 cao
    c['dimensions'] = dict(c['dimensions'], width=dim[0], height=dim[1], depth=dim[2], confidence=0.7)
    wp=np.array(pos,float); wR=eul(rot)
    if parent:
        pp,pR,ps=WORLD[parent]
        lp=(pR.T@(wp-pp))/ps
        lR=pR.T@wR
        # Euler XYZ từ ma trận
        sy=-lR[2,0] if False else lR[0,2]; ry=math.asin(max(-1,min(1,lR[0,2])))
        if abs(lR[0,2])<0.9999: rx=math.atan2(-lR[1,2],lR[2,2]); rz=math.atan2(-lR[0,1],lR[0,0])
        else: rx=math.atan2(lR[2,1],lR[1,1]); rz=0.0
        lrot=[rx,ry,rz]
        lsc=list(scale)
    else:
        lp=wp; lrot=list(rot); lsc=list(scale)
    WORLD[id]=(wp,wR,np.ones(3))
    c['transform'] = dict(position=[round(float(v),4) for v in lp], rotation=[round(float(v),4) for v in lrot])   # ¬ scale: dimensions bake vào geometry (¬ kế thừa)
    c['localFeatures'] = feats or []
    c['actionProfile']['pivot']['localPosition'] = [0, 0, 0]
    c['actionProfile']['animationRole'] = role
    if parent:
        c['attachment'] = dict(parentSocket=f'{parent}-socket', contactType='overlap', localStart=[0,0,0], localEnd=[0,0,0], embed=0.02, overlap=0.04, gapTolerance=0.02)
    else:
        c['attachment'] = None
    if extra: c['geometryDescriptor'].update(extra)
    comps.append(c); return c
add('root','Mouse Jester','macro','ellipsoid',None,[0,0,0],[0.1,0.1,0.1],'fur-black',role='root')
# torso + costume
add('torso','Torso','macro','ellipsoid','root',[0,2.25,0],[0.8,1.2,0.6],'fur-black',role='body')
add('ruff','Ruff collar','meso','torus','torso',[0,2.5,0.02],[1.0,1.0,0.3],'cloth-teal',rot=(math.pi/2,0,0))
add('ruff-trim','Ruff orange trim','micro','torus','ruff',[0,2.5,0.06],[1.1,1.1,0.12],'cloth-orange',rot=(math.pi/2,0,0))
add('bowtie-a','Bow-tie wing A','meso','cone','torso',[-0.22,1.97,0.32],[0.3,0.38,0.3],'cloth-purple',rot=(0,0,math.pi/2))
add('bowtie-b','Bow-tie wing B','meso','cone','torso',[0.22,1.97,0.32],[0.3,0.38,0.3],'cloth-purple',rot=(0,0,-math.pi/2))
add('skull','Skull charm','micro','sphere','torso',[0,1.97,0.38],[0.2,0.2,0.18],'bone')
add('bloomers','Harlequin bloomers','macro','ellipsoid','root',[0,1.3,0],[1.5,1.25,0.95],'cloth-teal',role='body')
add('belt-band','Waist band','meso','cylinder','bloomers',[0,1.85,0],[0.9,0.12,0.6],'cloth-purple')
add('suspender-a','Suspender A','meso','box','torso',[-0.18,2.15,0.34],[0.07,0.7,0.04],'cloth-orange')
add('suspender-b','Suspender B','meso','box','torso',[0.18,2.15,0.34],[0.07,0.7,0.04],'cloth-orange')
for sx,n in ((-1,'R'),(1,'L')):
    add(f'xbutton-{n}','X button '+n,'micro','cylinder','bloomers',[sx*0.18,1.78,0.43],[0.14,0.03,0.14],'cloth-orange',rot=(math.pi/2,0,0))
    add(f'diamond-{n}','Diamond print '+n,'micro','box','bloomers',[sx*0.35,1.0,0.48],[0.3,0.3,0.03],'cloth-orange',rot=(0,0,math.pi/4))
    # legs / stockings / shoes
    add(f'leg-{n}',f'Stocking {n}','meso','capsule','root',[sx*0.45,0.72,0.0],[0.2,0.5,0.2],'cloth-purple',role='leg')
    add(f'shoe-{n}',f'Curled shoe {n}','macro','ellipsoid','root',[sx*0.62,0.24,0.12],[0.95,0.5,0.75],'shoe-satin',role='foot')
    add(f'shoe-toe-{n}',f'Shoe toe curl {n}','meso','cone','shoe-'+n,[sx*0.75,0.3,0.22],[0.2,0.55,0.2],'cloth-orange',rot=(0,0,-sx*math.pi/2.8))
    add(f'shoe-bell-{n}',f'Shoe bell {n}','micro','sphere',f'shoe-toe-{n}',[sx*0.98,0.42,0.22],[0.12,0.12,0.12],'gold-metal')
# heads: (id, name, centre px, size w,h,d, hat kind)
heads = [('head-c','Centre head (jester)',(627,215),(1.0,0.9,0.85)), ('head-w','Left head (witch hat, wink)',(335,305),(0.95,0.85,0.8)), ('head-t','Right head (top hat, bored)',(925,305),(0.95,0.85,0.8))]
for hid,hn,(px,py),dim in heads:
    x,y = U(px,py)
    add(hid,hn,'macro','ellipsoid','torso',[x,y,0.12],dim,'face-cream',role='head')
    # ears (two per head)
    for sx,n in ((-1,'R'),(1,'L')):
        add(f'{hid}-ear-{n}',f'{hn} ear {n}','meso','cylinder',hid,[x+sx*0.5,y+0.32,-0.05],[0.42,0.06,0.42],'fur-black',rot=(math.pi/2,0,0))
        add(f'{hid}-eye-{n}',f'{hn} eye {n}','micro','sphere',hid,[x+sx*0.17,y+0.08,0.4],[0.11,0.13,0.05],'ink-white')
        add(f'{hid}-pupil-{n}',f'{hn} star pupil {n}','micro','sphere',f'{hid}-eye-{n}',[x+sx*0.17,y+0.08,0.44],[0.07,0.08,0.03],'cloth-orange')
    add(f'{hid}-nose',f'{hn} nose','micro','sphere',hid,[x,y-0.1,0.44],[0.1,0.07,0.06],'fur-black')
    add(f'{hid}-mouth',f'{hn} mouth','micro','ellipsoid',hid,[x,y-0.28,0.38],[0.3,0.16,0.08],'fur-black')
add('head-w-tongue','Left head tongue','micro','ellipsoid','head-w',[-1.0,2.55,0.45],[0.12,0.2,0.05],'tongue-pink')
# hats
cx,cy = U(627,60); add('hat-crown','Jester crown','meso','cone','head-c',[cx,cy,0.1],[0.5,0.45,0.4],'cloth-orange')
lx,ly = U(335,120); add('hat-witch','Witch/jingle hat','meso','cone','head-w',[lx,ly,0.05],[0.55,0.6,0.5],'cloth-purple',rot=(0,0,0.5))
rx,ry = U(930,150); add('hat-top','Top hat','meso','cylinder','head-t',[rx,ry,0.05],[0.4,0.3,0.4],'cloth-teal',rot=(0,0,-0.1)); add('hat-top-brim','Top-hat brim','micro','cylinder','hat-top',[rx,ry-0.14,0.05],[0.55,0.04,0.55],'cloth-teal',rot=(0,0,-0.1))
for bid,(bx,by) in (('bell-crown',(627,15)),('bell-witch',(275,105))): 
    x,y=U(bx,by); add(bid,bid,'micro','sphere','hat-crown' if 'crown' in bid else 'hat-witch',[x,y,0.1],[0.1,0.1,0.1],'gold-metal')
# arms (3 tiers per side): shoulder -> hand
tiers = [('up',(520,430),(150,400)),('mid',(510,470),(255,610)),('low',(505,500),(215,650))]
for sx,n in ((-1,'R'),(1,'L')):
    for t,(sp,hp) in zip(('up','mid','low'),[((520,430),(150,400)),((510,470),(255,610)),((505,505),(215,655))]):
        sp = (627 + sx*(627-sp[0]), sp[1]); hp_ = (627 + sx*(627-hp[0]), hp[1])
        a = U(*sp); b = U(*hp_)
        dx,dy = b[0]-a[0], b[1]-a[1]; L = math.hypot(dx,dy)
        add(f'arm-{t}-{n}',f'Arm {t} {n}','meso','capsule','torso',[round((a[0]+b[0])/2,3),round((a[1]+b[1])/2,3),0.0],[0.3,L,0.3],'fur-black',rot=(0,0,round(math.atan2(-dx,dy),3)),role='arm')
        add(f'hand-{t}-{n}',f'Hand {t} {n}','meso','sphere',f'arm-{t}-{n}',[b[0],b[1],0.0],[0.38,0.38,0.2],'fur-black',role='hand')
        add(f'cuff-{t}-{n}',f'Cuff scrunchie {t} {n}','micro','torus',f'arm-{t}-{n}',[round(b[0]-dx/L*0.18,3),round(b[1]-dy/L*0.18,3),0.0],[0.2,0.2,0.1],'cloth-orange' if t!='mid' else 'cloth-teal',rot=(0,0,round(math.atan2(-dx,dy),3)))
# props
bx,by = U(120,540); add('prop-ball','Star ball','meso','sphere','hand-low-R',[bx+0.05,by,0.05],[0.4,0.4,0.4],'cloth-purple')
cx2,cy2 = U(1130,590); add('prop-cards','Playing cards','meso','box','hand-low-L',[cx2-0.05,cy2,0.05],[0.45,0.5,0.03],'ink-white',rot=(0,0,-0.3))
# tails with charms
for sx,n in ((-1,'R'),(1,'L')):
    tail_pts = [[0.0,0.0,0.0],[sx*0.5,-0.1,-0.1],[sx*0.95,0.2,-0.15],[sx*1.25,0.5,-0.1]]
    c = add(f'tail-{n}',f'Tail {n}','meso','tube','root',[sx*0.25,1.35,-0.35],[1,1,1],'fur-black',role='tail',extra={'tubePath':{'points':tail_pts,'radius':0.04,'closed':False}})
    x,y = U(150 if sx<0 else 1100, 940)
    add(f'charm-{n}',f'Tail charm {n}','micro','sphere',f'tail-{n}',[sx*1.5,1.8,-0.3],[0.14,0.14,0.05],'gold-metal')
s['componentTree'] = comps
s['materials'] = mats
s['repetitionSystems'] = [dict(id='cuff-scrunchies', componentRefs=[c['id'] for c in comps if c['id'].startswith('cuff-')], count=6, rule='one orange/teal scrunchie at every wrist (6 arms)', pattern='per-limb')]
s['featureReviewTargets'] = [
 dict(id='three-heads', name='Three distinct heads/expressions/hats', tier='critical', passIds=['blockout','structural-pass','form-refinement'], minimumScore=0.75, mustPass=True, componentRefs=['head-c','head-w','head-t','hat-crown','hat-witch','hat-top'], evidenceRefs=['full-object']),
 dict(id='six-arms', name='Six arms in three tiers with gestures', tier='critical', passIds=['blockout','structural-pass'], minimumScore=0.75, mustPass=True, componentRefs=[c['id'] for c in comps if c['id'].startswith('arm-')], evidenceRefs=['full-object']),
 dict(id='harlequin-costume', name='Teal/orange/purple harlequin costume + ruff + bow-tie', tier='critical', passIds=['material-pass','form-refinement'], minimumScore=0.75, mustPass=True, componentRefs=['bloomers','ruff','bowtie-a','bowtie-b','skull'], evidenceRefs=['full-object']),
 dict(id='curled-shoes-tails', name='Curled shoes with bells + curled tails', tier='important', passIds=['structural-pass','form-refinement'], minimumScore=0.7, mustPass=False, componentRefs=['shoe-toe-L','shoe-toe-R','tail-L','tail-R'], evidenceRefs=['full-object'])]
s['referenceCamera'] = dict(s['referenceCamera'], positionHint=[0,1.85,6.5], fovDegrees=38)
json.dump(s, open('object-sculpt-spec.json','w'), indent=1, ensure_ascii=False)
print(len(comps),'components', len(mats),'materials')
