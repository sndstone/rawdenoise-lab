#!/usr/bin/env python3
"""Apply the frozen runs/nafnet_round/protocol.json decision rules to an eval + identity JSON pair."""
import argparse, json

SHADOW = ['native|0.0625|high', 'native|0.015625|high', 'native|0.0625|tails']
TIERS = {'a_w32_s4': 'A', 'a_w32_mid12': 'A', 'a_w48_s3': 'B', 'a_w48_s4': 'out', 'nafnet_w64': 'out'}

ap = argparse.ArgumentParser()
ap.add_argument('--eval', required=True); ap.add_argument('--identity')
ap.add_argument('--baseline', default='nafnet_s1'); ap.add_argument('--replicate', default='nafnet_s2')
ap.add_argument('--extra-floor', action='append', default=[], metavar='A=B', help='extra replicate pair for F_cap')
ap.add_argument('--out')
a = ap.parse_args()
d = json.load(open(a.eval))
comp = {(c['a'], c['slice'], c['metric']): c for c in d['comparisons']}
slices = list(dict.fromkeys(c['slice'] for c in d['comparisons']))
arms = list(dict.fromkeys(c['a'] for c in d['comparisons'] if c['a'] != a.replicate))
F = {s: abs(comp[(a.replicate, s, 'model_psnr')]['mean_delta_db']) for s in slices}
summ = {(r['model'], r['slice']): r for r in d['summary']}
for pair in a.extra_floor:  # F_cap = |arm_a - arm_b| from summary means
    x, y = pair.split('=')
    for s in slices:
        F[s] = max(F[s], abs(summ[(x, s)]['model_psnr']-summ[(y, s)]['model_psnr']))
out = dict(floor=F, arms={})
lines = ['| arm | tier | ' + ' | '.join(s.replace('native|', '').replace('0.0625', '1/16').replace('0.015625', '1/64') for s in slices) + ' | shadow mean | worst loss ok |',
         '|---|---|' + '---:|'*len(slices) + '---:|---|']
for arm in arms:
    row, cells, ok = {}, [], True
    for s in slices:
        c = comp.get((arm, s, 'model_psnr'))
        if not c: cells.append('—'); continue
        dlt, lo, hi = c['mean_delta_db'], *c['ci95']
        thr = max(2*F[s], .05)
        sig = abs(dlt) > thr and (lo > 0 or hi < 0)
        if dlt < -max(2*F[s], .10) and hi < 0: ok = False
        ci = comp.get((arm, s, 'interior_psnr'))
        row[s] = dict(delta=dlt, ci95=c['ci95'], significant=sig, interior_delta=ci and ci['mean_delta_db'],
                      sources_worse=c['sources_a_worse'])
        cells.append(f"{dlt:+.3f}{'*' if sig else ''}")
    sm = sum(row[s]['delta'] for s in SHADOW if s in row)/len(SHADOW)
    out['arms'][arm] = dict(tier=TIERS.get(arm, '-'), slices=row, shadow_mean=sm, no_excess_loss=ok)
    lines.append(f"| {arm} | {TIERS.get(arm, '-')} | " + ' | '.join(cells) + f" | {sm:+.3f} | {'yes' if ok else 'NO'} |")
lines.append('')
lines.append('`*` = |delta| > max(2F, 0.05) and 95% CI excludes 0. F per slice: ' + ', '.join(f'{F[s]:.3f}' for s in slices))
elig = [k for k, v in out['arms'].items() if v['tier'] in 'AB' and v['no_excess_loss']]
if elig:
    best = max(elig, key=lambda k: out['arms'][k]['shadow_mean'])
    out['capacity_ranking'] = sorted(elig, key=lambda k: -out['arms'][k]['shadow_mean'])
    lines.append(f"Capacity ranking (eligible): {', '.join(f'{k} {out['arms'][k]['shadow_mean']:+.3f}' for k in out['capacity_ranking'])}")
if a.identity:
    ident = json.load(open(a.identity))
    out['identity'] = {}
    lines += ['', '| arm | ' + ' | '.join(ident['kinds']) + ' | gate |', '|---|' + '---|'*(len(ident['kinds'])+1)]
    for m in ident['checkpoints']:
        rs = [r for r in ident['summary'] if r['model'] == m and r['path'] == 'tiled']
        out['identity'][m] = {r['kind']: dict(max=r['max_abs_error'], bias=r['mean_bias'], passed=r['passed']) for r in rs}
        lines.append(f"| {m} | " + ' | '.join(f"{r['max_abs_error']:.2g}{'' if r['passed'] else ' ✗'}" for r in rs) +
                     f" | {'PASS' if all(r['passed'] for r in rs) else 'FAIL'} |")
print('\n'.join(lines))
if a.out:
    json.dump(out, open(a.out, 'w'), indent=1)
    open(a.out.rsplit('.', 1)[0]+'.md', 'w').write('\n'.join(lines)+'\n')
