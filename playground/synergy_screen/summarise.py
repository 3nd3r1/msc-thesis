"""Pair each graph row in the screen log with its matched cosine row."""

import re
import sys
from pathlib import Path

ROW = re.compile(
    r"^\s*(\S+)\s+([\d,]+)\s+([\d.]+)%\s+(citation|bib-coupling|cosine k=\d+)\s+"
    r"([\d,]+)\s+([\d.]+)x\s+([\d.]+)x\s+([\d.]+)%\s+([\d.]+)%")

rows = []
for path in sys.argv[1:]:
    for line in Path(path).read_text().splitlines():
        m = ROW.match(line)
        if m:
            name, n, rate, graph, edges, lift, ctrl, nbr, reach = m.groups()
            rows.append(dict(name=name, n=int(n.replace(",", "")), rate=float(rate),
                             graph=graph, edges=int(edges.replace(",", "")),
                             lift=float(lift), ctrl=float(ctrl),
                             nbr=float(nbr), reach=float(reach)))

pairs = []
for i, r in enumerate(rows):
    if r["graph"].startswith("cosine"):
        continue
    nxt = rows[i + 1] if i + 1 < len(rows) else None
    if nxt and nxt["graph"].startswith("cosine") and nxt["name"] == r["name"]:
        pairs.append((r, nxt))

seen = {}
for g, c in pairs:
    seen.setdefault(g["name"], []).append((g, c))

print(f"{'review':>34} {'rows':>6} {'rate':>5} {'graph':>13} {'lift':>6} "
      f"{'cosine':>7} {'delta':>7} {'reach':>6}")
wins = {"citation": 0, "bib-coupling": 0}
total = {"citation": 0, "bib-coupling": 0}
best_wins = 0
for name, items in seen.items():
    deltas = []
    for g, c in items:
        d = g["lift"] - c["lift"]
        total[g["graph"]] += 1
        if d > 0:
            wins[g["graph"]] += 1
        deltas.append(d)
        print(f"{name:>34} {g['n']:>6,} {g['rate']:>4.1f}% {g['graph']:>13} "
              f"{g['lift']:>5.1f}x {c['lift']:>6.1f}x {d:>+6.1f}x {g['reach']:>5.1f}%")
    if max(deltas) > 0:
        best_wins += 1

print(f"\n{len(seen)} reviews screened")
for k in total:
    print(f"  {k:>13} beats matched cosine on {wins[k]} of {total[k]}")
print(f"  at least one graph wins on {best_wins} of {len(seen)} reviews")
won = [max(i, key=lambda x: x[0]['lift'] - x[1]['lift']) for i in seen.values()]
up = [g['lift'] - c['lift'] for g, c in won if g['lift'] > c['lift']]
down = [g['lift'] - c['lift'] for g, c in won if g['lift'] <= c['lift']]
print(f"  winning margin   mean {sum(up)/len(up):+.1f}x  max {max(up):+.1f}x")
print(f"  losing margin    mean {sum(down)/len(down):+.1f}x  worst {min(down):+.1f}x")

print()
for floor in (0, 30, 50, 80):
    ok = {k: v for k, v in seen.items() if v[0][0]["n"] * v[0][0]["rate"] / 100 >= floor}
    best = [max(i, key=lambda x: x[0]['lift'] - x[1]['lift']) for i in ok.values()]
    w = [g['lift'] - c['lift'] for g, c in best if g['lift'] > c['lift']]
    l = [g['lift'] - c['lift'] for g, c in best if g['lift'] <= c['lift']]
    print(f"positives >= {floor:>3}: {len(ok):>2} reviews, a graph wins on {len(w):>2}, "
          f"win mean {(sum(w)/len(w) if w else 0):+.1f}x, "
          f"loss mean {(sum(l)/len(l) if l else 0):+.1f}x")
