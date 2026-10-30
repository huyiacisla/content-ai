# Hậu xử lý spec: công thức màu, chi tiết nhỏ, detail inventory, ánh sáng, vật liệu cục bộ, anatomy
import json
s = json.load(open('object-sculpt-spec.json'))
def rgba(h): h=h.lstrip('#'); return f"rgba({int(h[0:2],16)}, {int(h[2:4],16)}, {int(h[4:6],16)}, 1.0)"
CLASS = {'fur-black':'fabric','face-cream':'skin','cloth-teal':'fabric','cloth-orange':'fabric','cloth-purple':'fabric','gold-metal':'metal','bone':'ceramic','tongue-pink':'skin','shoe-satin':'fabric','ink-white':'plastic'}
mats = {m['id']: m for m in s['materials']}
for c in s['componentTree']:
    m = mats[c['material']]
    c['colorMaterialRecipe'] = dict(dominantAlbedo=rgba(m['baseColor']), secondaryAlbedo=rgba(m['baseColor']), materialClass=CLASS[m['id']], materialClassConfidence=0.7,
        evidenceRef='reference.jpg flat toon fills (single front view)')
# vật liệu: roughness map độc lập, AO, local overrides, wear
for m in s['materials']:
    r = m['roughness']; r = r if isinstance(r, dict) else {'base': r}
    r.update(map='independent-procedural-field', variation=0.1, localResponse='slightly lower roughness on fold highlights, higher in creases'); m['roughness'] = r
    m['ambientOcclusion'] = dict(strength=0.5, source='cavity-bias at folds and under hats/ruff')
    m['textureResolution'] = max(1024, m.get('textureResolution', 1024))
mats['gold-metal']['localOverrides'] = [dict(id='bell-slot', region='bell lower slit', roughness=0.2, note='dark slit line on every bell')]
mats['cloth-teal']['localOverrides'] = [dict(id='diamond-seams', region='bloomers harlequin diamonds', note='darker teal diamond panels with purple stripe seams')]
mats['cloth-purple']['localOverrides'] = [dict(id='star-moon-print', region='purple skirt/hat', note='orange star + crescent print motifs')]
mats['fur-black']['wear'] = dict(amount=0.05, note='clean ink-black cartoon fills; faint cel-shade only')
comp = {c['id']: c for c in s['componentTree']}
def feat(cid, fid, kind, note): comp[cid]['localFeatures'].append(dict(id=fid, kind=kind, note=note))
feat('hat-top','spade-card','decal','spade playing-card badge on top-hat band')
feat('head-c','star-pupils','decal','orange star-shaped pupils both eyes')
feat('head-c','centre-fangs','contour','two small fangs in open grin')
feat('head-w','wink','contour','left eye closed in a wink arc with star-mark cheek')
feat('head-t','half-lid','contour','right head half-lidded bored eye + fang')
feat('bloomers','harlequin-diamonds','linework','teal diamond panels edged with purple/orange stripes')
feat('ruff','ruff-scallops','ridge','scalloped orange-edged ruff points')
feat('skull','skull-eyes','hole','two dark eye sockets + nose triangle on bone skull')
feat('xbutton-L','x-stitch','fastener','X-shaped stitching across orange buttons')
feat('shoe-bell-L','bell-slit','groove','slit across gold bell')
feat('tail-L','tail-charm-hook','contour','tail curls outward ending in a hanging crescent/star charm')
feat('prop-ball','ball-stars','decal','orange star print on purple juggling ball')
dets = [('d-star-pupils','decal','star-pupils','head-c'),('d-fangs','contour','centre-fangs','head-c'),('d-wink','contour','wink','head-w'),('d-halflid','contour','half-lid','head-t'),
 ('d-diamonds','linework','harlequin-diamonds','bloomers'),('d-ruff','ridge','ruff-scallops','ruff'),('d-skull','hole','skull-eyes','skull'),('d-xstitch','fastener','x-stitch','xbutton-L'),
 ('d-bell-slit','groove','bell-slit','shoe-bell-L'),('d-tail-charm','contour','tail-charm-hook','tail-L'),('d-ball-stars','decal','ball-stars','prop-ball'),('d-spade','decal','spade-card','hat-top'),
 ('d-bell-gloss','gloss','bell-slot','gold-metal')]
inv = s['preSpecAssessment']['detailInventory']
inv['details'] = [dict(id=i, kind=k, label=f, mapsTo=dict(ref=(f if f=='bell-slot' else f'{c}/{f}')), zone='full-object') for i,k,f,c in dets]
# gloss/bell-slot ref phải khớp khóa 'gold-metal/bell-slot'
inv['details'][-1]['mapsTo']['ref'] = 'gold-metal/bell-slot'
o = s['preSpecAssessment']['objectClass']
o.update(primaryType='stylized cartoon creature (three-headed six-armed mouse jester)', primaryDomain='hybrid', formLanguage=['organic','toon','curvilinear','bilateral'], structureKind=['multi-head','multi-limb','articulated','costume-layered'], motionPotential=['limb-gesture','head-bob','tail-sway','idle-breathe'], materialFamilies=['fabric','fur','metal','skin'])
a = s['preSpecAssessment']['anatomy']
a.update(applies=True, styleHeads=3.9, confidence=0.6, proportions=dict(headUnit=0.95, torso=1.2, legs=1.45, shoulderWidth=0.8, hipWidth=1.5), pose=dict(type='frontal A-pose with gestures', jointAngles={'armUpper':100,'armMid':60,'armLow':45}), faceLandmarks=dict(eyeLine=0.58, eyeSpacing=0.34, noseBase=0.42, mouthLine=0.28, hairline=0.95), features=['star pupils','fangs','whiskers'], note='non-human cartoon: proportions measured from reference px; no canon table')
for t in s['featureReviewTargets']:
    if t['id']=='harlequin-costume': t['id']='outfit-and-palette'
s['featureReviewTargets'].append(dict(id='face-landmark-placement', name='Eye/nose/mouth placement per head', tier='important', passIds=['form-refinement'], minimumScore=0.7, mustPass=False, componentRefs=['head-c','head-w','head-t'], evidenceRefs=['full-object']))
s['lightingFromPhoto'] = ['key: soft frontal-top directional (flat toon, intensity ~1.0), warm white', 'fill: hemisphere ambient 0.7 sky/ground cream', 'rim: faint back-light to separate black fur from background', 'exposure 1.0, ACES filmic tone mapping, sRGB output', 'contact shadow: soft ground shadow ellipse under shoes (ambient occlusion on floor)']
s['assumptions'] = list(s.get('assumptions') or []) + ['back side of body/heads is hidden — mirrored/inferred', 'hand finger shapes simplified to spheres', 'tail root attachment hidden behind bloomers']
s['preSpecAssessment']['unknownsToResolveBeforeImplementation'] = []
json.dump(s, open('object-sculpt-spec.json','w'), indent=1, ensure_ascii=False)
