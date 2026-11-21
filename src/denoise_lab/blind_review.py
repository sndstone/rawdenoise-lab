"""Offline blind comparisons. Keys contain identities; browser data never does."""
import hashlib
import itertools
import json
from pathlib import Path
import uuid

import numpy as np


def digest(path):
    """Hash files or a model package, including relative package filenames."""
    path = Path(path)
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    for file in files:
        if path.is_dir():
            h.update(str(file.relative_to(path)).encode())
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                h.update(chunk)
    return h.hexdigest()


def select_crops(rgb, count=5, size=256, seed=0):
    """Seeded unique windows, favouring mixed low/high gradient content on input.

    Rank a random candidate pool by gradient spread, then sample its upper half.
    No model output influences selection. Coordinates refer to oriented renders.
    """
    h, w = rgb.shape[:2]
    scale = np.iinfo(rgb.dtype).max if np.issubdtype(rgb.dtype, np.integer) else 1.
    if count < 1 or size < 8 or size > min(h, w):
        raise ValueError('Need count >= 1 and 8 <= crop size <= image dimensions')
    available = (h-size+1)*(w-size+1)
    if count > available:
        raise ValueError('Too many unique crops requested')
    rng = np.random.default_rng(seed)
    ids = rng.choice(available, min(available, max(count*16, 64)), replace=False)
    candidates = []
    for index in ids:
        y, x = divmod(int(index), w-size+1)
        patch = rgb[y:y+size:4, x:x+size:4].mean(-1)/scale
        gy, gx = np.gradient(patch)
        g = np.hypot(gx, gy)
        lo, hi = np.percentile(g, [25, 90])
        score = (hi-lo)/(lo+.01)
        candidates.append((score, y, x))
    candidates.sort(reverse=True)
    pool = candidates[:max(count, len(candidates)//2)]
    return [(pool[i][1], pool[i][2]) for i in rng.choice(len(pool), count, replace=False)]


def render(source, denoiser=None, *, strength=1., exposure_ev=0., prepass=True, overlap=60):
    """Render with fixed camera WB, sRGB primaries and no automatic exposure."""
    import rawpy
    from .bayer import PHASES, normalize_raw, pack, unpack
    from .dng import RawMetadata, sensor_codes
    with rawpy.imread(str(source)) as raw:
        meta = RawMetadata.from_raw(raw, source)
        info = dict(camera=f'{meta.make} {meta.model}', iso='unknown',
                    shutter=None, aperture=None, orientation=meta.orientation)
        # rawpy exposes no EXIF; TIFF-family RAWs may carry an EXIF IFD.
        import tifffile
        try:
            with tifffile.TiffFile(source) as tif:
                tags = tif.pages[0].tags
                exif = tags[34665].value if 34665 in tags else {}
                for field, tag, name in [('iso', 34855, 'ISOSpeedRatings'),
                                         ('shutter', 33434, 'ExposureTime'),
                                         ('aperture', 33437, 'FNumber')]:
                    value = tags[tag].value if tag in tags else exif.get(name)
                    if isinstance(value, tuple):
                        value = value[0]/value[1] if field != 'iso' and len(value) == 2 else value[0]
                    if value is not None:
                        info[field] = value
        except (tifffile.TiffFileError, OSError):
            pass
        flip = raw.sizes.flip
        if denoiser is not None:
            predicted = denoiser.denoise_packed(pack(normalize_raw(raw)), strength=strength,
                                               prepass=prepass, overlap=overlap)
            codes = raw.raw_image_visible.astype(np.float32)
            black = np.tile(meta.black, ((codes.shape[0]+1)//2, (codes.shape[1]+1)//2))[:codes.shape[0], :codes.shape[1]]
            white = np.tile(meta.white, ((codes.shape[0]+1)//2, (codes.shape[1]+1)//2))[:codes.shape[0], :codes.shape[1]]
            visible = (codes-black)/(white-black)
            y, x = PHASES[meta.phase]
            mosaic = unpack(predicted)
            visible[y:y+mosaic.shape[0], x:x+mosaic.shape[1]] = mosaic
            raw.raw_image_visible[:] = sensor_codes(visible, meta)
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16,
                              output_color=rawpy.ColorSpace.sRGB, gamma=(1, 1),
                              user_flip=0, adjust_maximum_thr=0.).astype(np.float32)/65535.
    rgb = {3: lambda a: np.rot90(a, 2), 5: lambda a: np.rot90(a),
           6: lambda a: np.rot90(a, -1)}.get(flip, lambda a: a)(rgb)
    rgb = np.clip(rgb*2.**exposure_ev, 0, 1)
    rgb = np.where(rgb <= .0031308, rgb*12.92, 1.055*rgb**(1/2.4)-.055)
    return np.rint(rgb*255).astype(np.uint8), info


def prepare(sources, arms, output, *, count=5, size=256, seed=0, exposure_ev=0.,
            device='cpu', prepass=True, overlap=60):
    """Arms are dictionaries with label, model path and strength. Output must be new."""
    from .inference import Denoiser
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError('Review preparation requires optional Pillow (included in the dev extra)') from exc
    if len(arms) < 2 or len({a['label'] for a in arms}) != len(arms):
        raise ValueError('Need at least two uniquely labelled arms')
    if not sources or not np.isfinite(exposure_ev):
        raise ValueError('Need sources and finite exposure')
    if any(not 0 <= a.get('strength', 1.) <= 1 for a in arms):
        raise ValueError('Arm strength must be in [0,1]')
    arms = [dict(a, model=str(Path(a['model']).resolve())) for a in arms]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'assets').mkdir()
    (output/'private').mkdir()
    rng = np.random.default_rng(seed)
    def token():
        return rng.bytes(16).hex()
    models = {}
    for arm in arms:
        path = str(Path(arm['model']).resolve())
        if path not in models:
            models[path] = Denoiser.load(path, device=device)
    key = dict(version=1, review_id=uuid.uuid4().hex, seed=seed,
               settings=dict(count=count, size=size, exposure_ev=exposure_ev, device=device,
                             prepass=prepass, overlap=overlap, selection='noisy mixed-gradient'),
               arms=[dict(a, sha256=digest(a['model'])) for a in arms], images=[], trials={})
    public = []
    def save(rgb):
        name = 'assets/'+token()+'.png'
        Image.fromarray(rgb).save(output/name)
        return name
    for image_index, source in enumerate(sources):
        overrides = source if isinstance(source, dict) else {}
        source = Path(overrides.get('path', source)).resolve()
        print(f'Rendering image {image_index+1}/{len(sources)}: {source.name}', flush=True)
        noisy, info = render(source, exposure_ev=exposure_ev)
        info.update({k: v for k, v in overrides.items() if k in {'iso', 'camera', 'scene', 'split', 'shutter', 'aperture'}})
        key['images'].append(dict(path=str(source), sha256=digest(source), **info))
        crops = select_crops(noisy, count, size, seed+image_index)
        noisy_assets = [save(noisy[y:y+size, x:x+size]) for y, x in crops]
        # Save one arm at a time to bound memory on full sensor images.
        assets, overviews = [], []
        for arm in arms:
            rgb, _ = render(source, models[str(Path(arm['model']).resolve())],
                            strength=arm.get('strength', 1.), exposure_ev=exposure_ev,
                            prepass=prepass, overlap=overlap)
            assets.append([save(rgb[y:y+size, x:x+size]) for y, x in crops])
            overview = Image.fromarray(rgb)
            overview.thumbnail((1200, 900))
            overviews.append(save(np.asarray(overview)))
        for crop_index, (y, x) in enumerate(crops):
            for a, b in itertools.combinations(range(len(arms)), 2):
                left, right = (a, b) if rng.integers(2) else (b, a)
                trial = token()
                key['trials'][trial] = dict(left=left, right=right, image=image_index,
                                            crop_yx=[y, x], size=size)
                public.append(dict(id=trial, left=assets[left][crop_index], right=assets[right][crop_index],
                                   noisy=noisy_assets[crop_index], full_left=overviews[left],
                                   full_right=overviews[right]))
    rng.shuffle(public)
    (output/'private'/'key.json').write_text(json.dumps(key, indent=2))
    data = json.dumps(dict(review_id=key['review_id'], trials=public)).replace('<', '\\u003c')
    (output/'index.html').write_text(HTML.replace('__DATA__', data))
    return key


def decode_votes(key, votes):
    """Reject mismatched sessions, unknown trials and invalid choices; allow partial reviews."""
    if votes.get('review_id') != key['review_id']:
        raise ValueError('Votes belong to a different review')
    rows = []
    for trial_id, choice in votes['votes'].items():
        if trial_id not in key['trials'] or type(choice) is not int or choice not in (0, 1, 2):
            raise ValueError('Unknown trial or invalid vote')
        t = key['trials'][trial_id]
        rows.append((t['left'], t['right'], {0: .5, 1: 1., 2: 0.}[choice], t['image']))
    return rows


def bradley_terry(rows, n):
    """Mean-zero log strengths; ties count as half wins, ridge .01 keeps separation finite."""
    design = np.zeros((len(rows), n))
    outcomes = np.array([r[2] for r in rows])
    for k, (a, b, *_rest) in enumerate(rows):
        design[k, a], design[k, b] = 1, -1
    beta = np.zeros(n)
    for _ in range(100):
        p = 1/(1+np.exp(-np.clip(design@beta, -30, 30)))
        gradient = design.T@(outcomes-p)-.01*beta
        hessian = (design.T*(p*(1-p)))@design+.01*np.eye(n)
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-7:
            break
    return beta-beta.mean()


def score(key, votes, output, *, bootstrap=1000, seed=0):
    """Image-cluster percentile bootstrap respects correlated crops from each photo."""
    rows = decode_votes(key, votes)
    if not rows or bootstrap < 1:
        raise ValueError('Need votes and at least one bootstrap replicate')
    n = len(key['arms'])
    strengths = bradley_terry(rows, n)
    images = sorted({r[3] for r in rows})
    clusters = {i: [r for r in rows if r[3] == i] for i in images}
    rng = np.random.default_rng(seed)
    samples = [bradley_terry([r for i in rng.choice(images, len(images)) for r in clusters[i]], n)
               for _ in range(bootstrap)]
    low, high = np.percentile(samples, [2.5, 97.5], axis=0)
    labels = [a['label'].replace('|', '\\|').replace('\n', ' ') for a in key['arms']]
    lines = ['# Blind review results', '', f"Votes: {len(rows)}/{len(key['trials'])}; images: {len(images)}.", '',
             'Bradley–Terry mean-zero log strengths (ties = half wins; ridge = 0.01).',
             f'95% percentile CIs from {bootstrap} image-cluster bootstrap replicates, seed {seed}.',
             'Small image counts, disconnected comparisons and incomplete votes limit inference.', '',
             '| Model | Strength | 95% CI |', '|---|---:|---|']
    for i in np.argsort(-strengths):
        lines.append(f'| {labels[i]} | {strengths[i]:.3f} | [{low[i]:.3f}, {high[i]:.3f}] |')
    def table(title, subset):
        lines.extend(['', f'## {title}', '', '| Model A | Model B | A wins | Ties | A losses |', '|---|---|---:|---:|---:|'])
        for a, b in itertools.combinations(range(n), 2):
            outcomes = [v if i == a else 1-v for i, j, v, _ in subset if {i, j} == {a, b}]
            lines.append(f'| {labels[a]} | {labels[b]} | {outcomes.count(1.)} | {outcomes.count(.5)} | {outcomes.count(0.)} |')
    table('All pairs', rows)
    for i in images:
        table(f'Image {i+1}: {Path(key["images"][i]["path"]).name}', clusters[i])
    isos = sorted({str(key['images'][i]['iso']) for i in images})
    for iso in isos:
        table(f'ISO {iso}', [r for r in rows if str(key['images'][r[3]]['iso']) == iso])
    lines.extend(['', 'This report is visual evidence only; it does not establish model acceptance.'])
    Path(output).write_text('\n'.join(lines)+'\n')
    return strengths


HTML = '''<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind visual review</title>
<style>
body{background:#202020;color:#eee;font:16px system-ui;margin:20px}button{padding:10px;margin:4px}
.pair{display:flex;gap:16px;overflow:auto}.pane{flex:0 0 auto}.pane img{display:block;max-width:none;image-rendering:pixelated}
progress{width:100%}#noisy{overflow:auto}#status{min-height:24px}
</style>
<h1>Blind visual review</h1>
<p>1: left better · 2: right better · 0: tie · Space: next · n: noisy · z: 1:1 / 2× · f: overview</p>
<p>Judge detail, colour, artifacts and noise. A vote advances automatically; use Previous to revise.</p>
<progress id="progress"></progress><p id="status"></p>
<div><button id="prev">Previous</button><button id="leftVote">1 — Left</button><button id="tieVote">0 — Tie</button><button id="rightVote">2 — Right</button><button id="next">Next</button></div>
<div><button id="noiseButton">Noisy (n)</button><button id="zoomButton">Zoom (z)</button><button id="fullButton">Overview (f)</button><button id="exportButton">Export votes</button></div>
<div class="pair"><div class="pane">Left<img id="left" alt="Left arm"></div><div class="pane">Right<img id="right" alt="Right arm"></div></div>
<div id="noisy" hidden>Noisy input<img id="noise" alt="Noisy input"></div>
<script>
'use strict';
const data=__DATA__, storageKey='blind-review-'+data.review_id;
let votes={}, index=0, zoom=1, full=false, storageOK=true;
const el=id=>document.getElementById(id);
try {const saved=JSON.parse(localStorage.getItem(storageKey)||'null');if(saved){votes=saved.votes||{};index=saved.index||0;}} catch(e){storageOK=false;}
index=Math.max(0,Math.min(index,data.trials.length-1));
function persist(){try{localStorage.setItem(storageKey,JSON.stringify({votes,index}));}catch(e){storageOK=false;}}
function show(){const t=data.trials[index];
 for(const side of ['left','right']){const im=el(side);im.src=t[full?'full_'+side:side];im.onload=()=>{im.width=im.naturalWidth*(full?1:zoom);};}
 el('noise').src=t.noisy;el('noise').onload=()=>{el('noise').width=el('noise').naturalWidth*zoom;};
 el('noisy').hidden=full||!noiseVisible;
 el('progress').max=data.trials.length;el('progress').value=Object.keys(votes).length;
 el('status').textContent=`Pair ${index+1}/${data.trials.length}; ${Object.keys(votes).length} voted; ${full?'overview':zoom+'×'}; current: ${votes[t.id]===undefined?'unvoted':({0:'tie',1:'left',2:'right'})[votes[t.id]]}`+(storageOK?'':' — Local saving unavailable: export before closing.');}
let noiseVisible=false;
function move(d){index=Math.max(0,Math.min(data.trials.length-1,index+d));persist();show();}
function vote(v){votes[data.trials[index].id]=v;move(1);}
el('prev').onclick=()=>move(-1);el('next').onclick=()=>move(1);
el('leftVote').onclick=()=>vote(1);el('rightVote').onclick=()=>vote(2);el('tieVote').onclick=()=>vote(0);
el('noiseButton').onclick=()=>{noiseVisible=!noiseVisible;show();};
el('zoomButton').onclick=()=>{full=false;zoom=zoom===1?2:1;show();};
el('fullButton').onclick=()=>{full=!full;show();};
el('exportButton').onclick=()=>{const blob=new Blob([JSON.stringify({version:1,review_id:data.review_id,votes},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='votes.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
document.addEventListener('keydown',e=>{
 if(['0','1','2'].includes(e.key)){e.preventDefault();vote(Number(e.key));}
 else if(e.key===' '){e.preventDefault();move(1);}
 else if(e.key.toLowerCase()==='n')el('noiseButton').click();
 else if(e.key.toLowerCase()==='z')el('zoomButton').click();
 else if(e.key.toLowerCase()==='f')el('fullButton').click();});
show();
</script>
'''
