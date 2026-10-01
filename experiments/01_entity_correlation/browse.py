import html
import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

import run
from data import load


RESULTS = Path(__file__).resolve().parent / "results"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
LIMIT = 100

SHORT = {
    "reports skin irritation or an allergic reaction": "irritation",
    "says the product doesn't work as advertised": "doesn't work",
    "suspects the product is fake": "fake",
    "mentions having sensitive skin": "sensitive skin",
    "bought it as a gift": "gift",
    "mentions another person (partner, child, friend)": "other person",
    "the review is positive": "positive",
}
PREDICATES = list(SHORT.values())
GROUPINGS = {
    "product": "parent_asin",
    "cluster": "cluster",
    "user": "user_id",
    "rating": "rating",
}

CSS = """
body { font: 15px/1.5 system-ui, sans-serif; max-width: 1100px; margin: 2rem auto;
       padding: 0 1rem; color: #111 }
a { color: #007185; text-decoration: none }
a:hover { text-decoration: underline }
h1 { font-size: 1.4rem; margin-bottom: .2rem }
h2 { font-size: 1.1rem; margin: 1.4rem 0 .3rem }
.muted { color: #666; font-size: .85rem }
.stars { color: #de7921; letter-spacing: 1px }
.review { border-top: 1px solid #ddd; padding: .8rem 0 }
.title { font-weight: 600 }
.chip { display: inline-block; background: #eef4f6; border: 1px solid #cdd; color: #246;
        border-radius: 3px; padding: 0 .35rem; margin: .15rem .2rem 0 0; font-size: .75rem }
table { border-collapse: collapse; width: 100%; font-size: .85rem }
td, th { text-align: left; padding: .3rem .5rem; border-bottom: 1px solid #eee;
         white-space: nowrap }
th { font-size: .78rem; color: #666; font-weight: 600 }
th a { color: #666 }
input { padding: .4rem; width: 24rem; border: 1px solid #bbb; border-radius: 3px }
.hbar { background: #de7921; height: 10px; display: inline-block }
.opts { margin: .4rem 0; font-size: .85rem }
.opts .label { color: #666; margin-right: .3rem }
.opts a { display: inline-block; padding: 0 .4rem; border: 1px solid #ddd;
          border-radius: 3px; margin-right: .2rem; color: #007185 }
.opts a.on { background: #232f3e; border-color: #232f3e; color: #fff }
"""


def esc(x):
    return html.escape("" if x is None or pd.isna(x) else str(x))


def page(title, body):
    return (
        f"<!doctype html><meta charset=utf-8><title>{esc(title)}</title>"
        f"<style>{CSS}</style>"
        f"<p><a href='/'>reviews</a> &nbsp; <a href='/groups'>groupings</a></p>{body}"
    )


def stars(rating):
    n = int(rating or 0)
    return f"<span class=stars>{'★' * n}{'☆' * (5 - n)}</span>"


def link(path, params, **changes):
    q = {k: v for k, v in {**params, **changes}.items() if v}
    return esc(path + ("?" + urllib.parse.urlencode(q) if q else ""))


def opts(label, path, params, param, choices):
    out = [f"<span class=label>{esc(label)}</span>"]
    for value, text in choices:
        on = " on" if params.get(param, "") == value else ""
        out.append(f"<a class='{on.strip() or ''}' href='{link(path, params, **{param: value}, page=None)}'>{esc(text)}</a>")
    return f"<div class=opts>{''.join(out)}</div>"


def header(path, params, columns):
    cells = []
    for col, text in columns:
        asc = "1" if params.get("sort") == col and not params.get("asc") else None
        mark = "" if params.get("sort") != col else (" ▲" if params.get("asc") else " ▼")
        cells.append(
            f"<th><a href='{link(path, params, sort=col, asc=asc)}'>{esc(text)}{mark}</a></th>"
        )
    return "<tr>" + "".join(cells) + "</tr>"


def order(df, params, default):
    col = params.get("sort") or default
    if col not in df.columns:
        col = default
    return df.sort_values(col, ascending=bool(params.get("asc")), na_position="last")


def search_form(q=""):
    return (
        "<form action=/search><input name=q placeholder='product id, user id or text' "
        f"value='{esc(q)}'></form>"
    )


def pager(path, params, shown, total):
    if total <= shown:
        return ""
    n = int(params.get("page", 1))
    bits = []
    if n > 1:
        bits.append(f"<a href='{link(path, params, page=str(n - 1))}'>previous</a>")
    if n * LIMIT < total:
        bits.append(f"<a href='{link(path, params, page=str(n + 1))}'>next</a>")
    return f"<div class=opts><span class=label>page {n}</span>{' '.join(bits)}</div>"


def cut(df, params):
    n = int(params.get("page", 1))
    return df.iloc[(n - 1) * LIMIT : n * LIMIT]


class Browser:
    def __init__(self):
        print("loading reviews")
        self.df = load()
        self.df["length"] = self.df.text.fillna("").str.len()
        self.products = self.df.groupby("parent_asin").indices
        self.users = self.df.groupby("user_id").indices
        self.counts = self.df.groupby("parent_asin").size()

        sample = pd.read_json(RESULTS / "sample.jsonl", lines=True)
        labels = pd.read_json(RESULTS / "labels.jsonl", lines=True)
        labels["short"] = labels.predicate.map(SHORT)
        sample = sample.join(labels.pivot(index="row", columns="short", values="verdict")[PREDICATES])
        self.sample = sample
        self._clusters = None
        self._stats = {}
        verdicts = {
            row: set(g.short[g.verdict == "yes"]) for row, g in labels.groupby("row")
        }
        self.labelled = {
            (r.parent_asin, r.user_id, str(r.text)): verdicts.get(i, set())
            for i, r in sample.iterrows()
        }

        lab = labels.join(sample[["parent_asin"]], on="row")
        rates = lab.pivot_table(
            index="parent_asin",
            columns="short",
            values="verdict",
            aggfunc=lambda v: (v == "yes").mean(),
        )
        self.summary = pd.DataFrame(
            {
                "reviews": self.counts.reindex(sample.parent_asin.unique()),
                "labelled": sample.groupby("parent_asin").size(),
                "bucket": sample.groupby("parent_asin").bucket.first(),
            }
        ).join(rates[PREDICATES])
        print(f"{len(self.df):,} reviews, {len(self.products):,} products")

    def clusters(self):
        if self._clusters is None:
            path = RESULTS / "clusters.json"
            if path.exists():
                self._clusters = json.loads(path.read_text())
            else:
                print("embedding the sample, this takes a minute")
                k = self.sample.parent_asin.nunique()
                self._clusters = [int(c) for c in run.embed_clusters(self.sample, k)]
                path.write_text(json.dumps(self._clusters))
            self.sample["cluster"] = self._clusters

    def key(self, by):
        if by == "cluster":
            self.clusters()
        return self.sample[GROUPINGS[by]]

    def stats(self, by):
        if by not in self._stats:
            keys = self.key(by)
            rng = np.random.default_rng(run.SEED)
            self._stats[by] = {
                p: run.kappa(keys, self.sample[p], None, rng) for p in PREDICATES
            }
        return self._stats[by]

    def yes_for(self, row):
        return self.labelled.get((row.parent_asin, row.user_id, str(row.text)))

    def chips(self, row):
        got = self.yes_for(row)
        if got is None:
            return ""
        return "".join(f"<span class=chip>{esc(y)}</span>" for y in sorted(got)) or (
            "<span class=muted>labelled, nothing fired</span>"
        )

    def review(self, row, show="user"):
        ref = (
            f"<a href='/user/{esc(row.user_id)}'>{esc(row.user_id[:12])}…</a>"
            if show == "user"
            else f"<a href='/product/{esc(row.parent_asin)}'>{esc(row.parent_asin)}</a>"
        )
        return (
            f"<div class=review>{stars(row.rating)} <span class=title>{esc(row.title)}"
            f"</span><div class=muted>{ref}</div><p>{esc(row.text)}</p>"
            f"{self.chips(row)}</div>"
        )

    def index(self, params):
        path = "/"
        scope = params.get("scope", "sampled")
        bar = opts(
            "show",
            path,
            params,
            "scope",
            [("sampled", "labelled products"), ("all", "all products")],
        )
        if scope == "all":
            table = pd.DataFrame({"reviews": self.counts})
            columns = [("reviews", "reviews")]
        else:
            table = self.summary
            bucket = params.get("bucket", "")
            bar += opts(
                "bucket",
                path,
                params,
                "bucket",
                [("", "any"), ("5-9", "5-9"), ("10-49", "10-49"), ("50+", "50+")],
            )
            if bucket:
                table = table[table.bucket == bucket]
            columns = [("reviews", "reviews"), ("labelled", "labelled")] + [
                (p, p) for p in PREDICATES
            ]

        table = order(table, params, "reviews")
        shown = cut(table, params)
        head = header(path, params, [("", "product")] + columns)
        body = ""
        for asin, r in shown.iterrows():
            cells = "".join(
                f"<td>{r[c]:.0%}</td>"
                if c in PREDICATES
                else f"<td>{0 if pd.isna(r[c]) else int(r[c]):,}</td>"
                for c, _ in columns
            )
            body += f"<tr><td><a href='/product/{esc(asin)}'>{esc(asin)}</a></td>{cells}</tr>"

        return page(
            "browse",
            f"<h1>All_Beauty</h1><p class=muted>{len(self.df):,} reviews, "
            f"{len(self.products):,} products, {len(self.users):,} users</p>"
            f"{search_form()}{bar}"
            f"<p class=muted>{len(table):,} products, columns are the share of "
            "labelled reviews where the predicate fired</p>"
            f"<table>{head}{body}</table>{pager(path, params, len(shown), len(table))}",
        )

    def product(self, asin, params):
        idx = self.products.get(asin)
        if idx is None:
            return page("not found", "<p>no such product</p>")
        path = f"/product/{asin}"
        rows = self.df.iloc[idx]
        counts = rows.rating.value_counts().reindex([5, 4, 3, 2, 1]).fillna(0)
        hist = "".join(
            f"<tr><td>{int(r)} star</td><td><span class=hbar "
            f"style='width:{n / len(rows) * 300:.0f}px'></span> {int(n)}</td></tr>"
            for r, n in counts.items()
        )
        head = (
            f"<h1>{esc(asin)}</h1><p class=muted>{len(rows):,} reviews, "
            f"mean rating {rows.rating.mean():.2f}</p><table>{hist}</table>"
        )

        bar = opts(
            "stars",
            path,
            params,
            "star",
            [("", "any")] + [(str(s), f"{s}") for s in [5, 4, 3, 2, 1]],
        )
        bar += opts(
            "predicate",
            path,
            params,
            "pred",
            [("", "any")] + [(p, p) for p in PREDICATES],
        )
        bar += opts(
            "sort",
            path,
            params,
            "sort",
            [("rating", "rating"), ("length", "length")],
        )
        bar += opts("order", path, params, "asc", [("", "high first"), ("1", "low first")])

        if params.get("star"):
            rows = rows[rows.rating == float(params["star"])]
        pred = params.get("pred")
        if pred in PREDICATES:
            keep = [pred in (self.yes_for(r) or set()) for r in rows.itertuples()]
            rows = rows[keep]
        rows = order(rows, params, "rating")
        shown = cut(rows, params)
        return page(
            asin,
            head
            + bar
            + f"<p class=muted>{len(rows):,} of {len(self.df.iloc[idx]):,} reviews shown</p>"
            + "".join(self.review(r) for r in shown.itertuples())
            + pager(path, params, len(shown), len(rows)),
        )

    def user(self, user_id, params):
        idx = self.users.get(user_id)
        if idx is None:
            return page("not found", "<p>no such user</p>")
        path = f"/user/{user_id}"
        rows = order(self.df.iloc[idx], params, "rating")
        shown = cut(rows, params)
        bar = opts("sort", path, params, "sort", [("rating", "rating"), ("length", "length")])
        bar += opts("order", path, params, "asc", [("", "high first"), ("1", "low first")])
        return page(
            user_id,
            f"<h1>{esc(user_id)}</h1><p class=muted>{len(rows):,} reviews, "
            f"mean rating {rows.rating.mean():.2f}</p>{bar}"
            + "".join(self.review(r, show="product") for r in shown.itertuples())
            + pager(path, params, len(shown), len(rows)),
        )

    def groups(self, params):
        path = "/groups"
        by = params.get("by", "product")
        if by not in GROUPINGS:
            by = "product"
        keys = self.key(by)
        bar = opts("group by", path, params, "by", [(b, b) for b in GROUPINGS])

        stats = self.stats(by)
        rows = "".join(
            f"<tr><td>{esc(p)}</td><td>{o:.2f}</td><td>{c:.2f}</td><td>{k:.2f}</td></tr>"
            for p, (o, c, k) in stats.items()
        )
        table = pd.DataFrame({"size": self.sample.groupby(keys).size()})
        for pred in PREDICATES:
            table[pred] = self.sample.groupby(keys)[pred].apply(
                lambda v: (v == "yes").mean()
            )

        minimum = params.get("min", "")
        bar += opts(
            "min size",
            path,
            params,
            "min",
            [("", "any"), ("2", "2+"), ("5", "5+"), ("10", "10+"), ("20", "20+")],
        )
        if minimum:
            table = table[table["size"] >= int(minimum)]

        table = order(table, params, "size")
        shown = cut(table, params)
        head = header(path, params, [("", by), ("size", "size")] + [(x, x) for x in PREDICATES])
        body = ""
        for value, r in shown.iterrows():
            cells = "".join(f"<td>{r[x]:.0%}</td>" for x in PREDICATES)
            body += (
                f"<tr><td><a href='/group/{by}/{urllib.parse.quote(str(value))}'>"
                f"{esc(value)}</a></td><td>{int(r['size']):,}</td>{cells}</tr>"
            )

        return page(
            f"groups by {by}",
            f"<h1>Groupings</h1><p class=muted>over the {len(self.sample):,} labelled "
            f"reviews of experiment 01, {len(self.sample.groupby(keys)):,} groups, "
            f"mean size {len(self.sample) / table['size'].count():.1f}</p>{bar}"
            "<h2>Agreement inside groups</h2>"
            "<p class=muted>obs is the share of same-group pairs that got the same "
            "verdict, chance is the same after shuffling verdicts across rows, "
            "k is (obs - chance) / (1 - chance)</p>"
            f"<table><tr><th>predicate<th>obs<th>chance<th>k</tr>{rows}</table>"
            f"<h2>Groups</h2><p class=muted>{len(table):,} groups, columns are the "
            "share of the group where the predicate fired</p>"
            f"<table>{head}{body}</table>{pager(path, params, len(shown), len(table))}",
        )

    def group(self, by, value, params):
        if by not in GROUPINGS:
            return page("not found", "<p>no such grouping</p>")
        keys = self.key(by)
        rows = self.sample[keys.astype(str) == value]
        if not len(rows):
            return page("not found", "<p>no such group</p>")
        path = f"/group/{by}/{urllib.parse.quote(value)}"
        rates = "".join(
            f"<tr><td>{esc(p)}</td><td>{(rows[p] == 'yes').mean():.0%}</td>"
            f"<td>{int((rows[p] == 'yes').sum())} of {len(rows)}</td></tr>"
            for p in PREDICATES
        )
        extra = (
            f" &nbsp; <a href='/product/{esc(value)}'>all {self.counts.get(value, 0):,} "
            "reviews of this product</a>"
            if by == "product"
            else ""
        )
        bar = opts("sort", path, params, "sort", [("rating", "rating"), ("length", "length")])
        bar += opts("order", path, params, "asc", [("", "high first"), ("1", "low first")])
        rows_sorted = order(rows.assign(length=rows.text.fillna("").str.len()), params, "rating")
        return page(
            f"{by} {value}",
            f"<h1>{esc(by)} {esc(value)}</h1><p class=muted>{len(rows):,} labelled "
            f"reviews, mean rating {rows.rating.mean():.2f}{extra}</p>"
            f"<table><tr><th>predicate<th>share<th>count</tr>{rates}</table>{bar}"
            + "".join(self.review(r, show="product") for r in rows_sorted.itertuples()),
        )

    def search(self, q, params):
        if q in self.products:
            return self.product(q, params)
        if q in self.users:
            return self.user(q, params)
        path = "/search"
        hit = self.df[
            self.df.title.fillna("").str.contains(q, case=False, regex=False)
            | self.df.text.fillna("").str.contains(q, case=False, regex=False)
        ]
        hit = order(hit, params, "rating")
        shown = cut(hit, params)
        bar = opts("sort", path, params, "sort", [("rating", "rating"), ("length", "length")])
        bar += opts("order", path, params, "asc", [("", "high first"), ("1", "low first")])
        return page(
            f"search {q}",
            f"{search_form(q)}{bar}<p class=muted>{len(hit):,} reviews match</p>"
            + "".join(self.review(r, show="product") for r in shown.itertuples())
            + pager(path, params, len(shown), len(hit)),
        )


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        parts = url.path.strip("/").split("/")
        params = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
        b = self.server.browser
        if parts == [""]:
            body = b.index(params)
        elif parts[0] == "product" and len(parts) > 1:
            body = b.product(urllib.parse.unquote(parts[1]), params)
        elif parts[0] == "user" and len(parts) > 1:
            body = b.user(urllib.parse.unquote(parts[1]), params)
        elif parts[0] == "groups":
            body = b.groups(params)
        elif parts[0] == "group" and len(parts) > 2:
            body = b.group(parts[1], urllib.parse.unquote(parts[2]), params)
        elif parts[0] == "search":
            body = b.search(params.pop("q", ""), params)
        else:
            body = page("not found", "<p>nothing here</p>")
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    server.browser = Browser()
    print(f"http://127.0.0.1:{PORT}")
    server.serve_forever()
