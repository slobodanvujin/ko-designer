"""
Mammalian KO Designer — CRISPR knockout strategy, guide design and a sequence map.
Paste this whole file into https://streamlit.io/playground and it runs as-is.
Needs only streamlit + pandas.

WHAT IT DOES THAT THE BIG TOOLS DON'T:
  - reasons about WHERE to cut (CDS position, exon, reinitiation risk) and explains why
  - finds downstream in-frame ATGs that could rescue a truncated protein
  - estimates whether microhomology-driven repair tends to give in-frame deletions
  - draws your sequence as a map, with the guides you tick shown where they bind
WHAT IT CANNOT DO:
  - genome-wide off-target search (needs an indexed genome) -> use CRISPOR/CHOPCHOP
  - published on-target models (Doench Rule Set 2 etc.) -> use CRISPOR/CHOPCHOP
"""
import re
import html
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

st.set_page_config(page_title="Mammalian KO Designer", page_icon="✂️", layout="wide")

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
CODONS = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}
VECTORS = {
    "lentiCRISPRv2 / lentiGuide-Puro (BsmBI)": ("BsmBI", "CACC", "AAAC"),
    "pSpCas9(BB)-2A-GFP / pX458, pX330, pX459 (BbsI)": ("BbsI", "CACC", "AAAC"),
    "pLKO5.sgRNA / U6 generic (BsmBI)": ("BsmBI", "CACC", "AAAC"),
}


# ------------------------------------------------------------------ sequence helpers
def rc(s):
    return s.translate(COMPLEMENT)[::-1]


def clean(s):
    return re.sub(r"[^ACGTNacgtn]", "", s).upper()


def parse_fasta_one(text):
    text = text.strip()
    if not text:
        return "", ""
    if text.startswith(">"):
        lines = text.splitlines()
        return lines[0][1:].strip() or "sequence", clean("".join(lines[1:]))
    return "sequence", clean(text)


def gc(s):
    return 100 * (s.count("G") + s.count("C")) / len(s) if s else 0.0


def translate(s):
    return "".join(CODONS.get(s[i:i + 3], "X") for i in range(0, len(s) - 2, 3))


def max_self_comp(s, min_len=4):
    r = rc(s)
    best = ""
    for i in range(len(s)):
        for j in range(i + min_len, len(s) + 1):
            if s[i:j] in r and j - i > len(best):
                best = s[i:j]
    return best


# ------------------------------------------------------------------ guide finding
def find_guides(seq, nuclease="SpCas9 (NGG)", spacer_len=20):
    """Guides with genomic (seq-coordinate) spans for the spacer, the PAM and the cut."""
    out, L = [], len(seq)
    for strand in ("+", "-"):
        s = seq if strand == "+" else rc(seq)
        if nuclease.startswith("SpCas9"):
            pam_re, pam_len, cut_off, slen = r"(?=[ACGT]GG)", 3, spacer_len - 3, spacer_len
        elif nuclease.startswith("SaCas9"):
            pam_re, pam_len, cut_off, slen = r"(?=[ACGT]{2}G[AG]{2}T)", 6, spacer_len - 3, spacer_len
        else:
            pam_re, pam_len, cut_off, slen = r"(?=TTT[ACG])", 4, None, 23
        for m in re.finditer(pam_re, s):
            p = m.start()
            if cut_off is not None:            # Cas9: spacer 5' of PAM
                a = p - slen
                if a < 0:
                    continue
                spacer, pam = s[a:p], s[p:p + pam_len]
                sp_lo, sp_hi, pam_lo, pam_hi = a, p, p, p + pam_len
                cut_local = a + cut_off
            else:                              # Cas12a: spacer 3' of PAM
                a = p + pam_len
                spacer, pam = s[a:a + slen], s[p:p + pam_len]
                sp_lo, sp_hi, pam_lo, pam_hi = a, a + slen, p, p + pam_len
                cut_local = a + 18
            if len(spacer) < slen or len(pam) < pam_len or "N" in spacer:
                continue

            def conv(lo, hi):
                return (lo, hi) if strand == "+" else (L - hi, L - lo)

            sp_lo, sp_hi = conv(sp_lo, sp_hi)
            pam_lo, pam_hi = conv(pam_lo, pam_hi)
            out.append({"strand": strand, "spacer": spacer, "pam": pam,
                        "sp_lo": sp_lo, "sp_hi": sp_hi, "pam_lo": pam_lo, "pam_hi": pam_hi,
                        "cut": cut_local if strand == "+" else L - cut_local})
    seen, uniq = set(), []
    for g in sorted(out, key=lambda x: x["cut"]):
        k = (g["spacer"], g["cut"])
        if k not in seen:
            seen.add(k)
            uniq.append(g)
    return uniq


def spliced_align(g, m):
    """Recover exon blocks by aligning an mRNA/CDS (m) onto its genomic sequence (g).
    Exact matching only — both must come from the same assembly. Returns (exons, error)."""
    exons, pm, pg = [], 0, 0
    while pm < len(m):
        rem, hit = len(m) - pm, None
        for sl in (26, 22, 18, 14, 11, 9):
            if sl > rem:
                continue
            idx = g.find(m[pm:pm + sl], pg)
            if idx != -1:
                hit = (idx, sl)
                break
        if hit is None:
            if rem <= 8 and exons:
                break
            return None, (f"Could not place mRNA base {pm + 1:,} in the genomic sequence. "
                          "The two sequences may be from different transcripts or assemblies, "
                          "or one may be reverse-complemented.")
        idx, sl = hit
        k = sl
        while pm + k < len(m) and idx + k < len(g) and m[pm + k] == g[idx + k]:
            k += 1
        if exons and idx <= exons[-1][1]:
            exons[-1] = (exons[-1][0], max(exons[-1][1], idx + k))
        else:
            exons.append((idx, idx + k))
        pm += k
        pg = idx + k
    if not exons:
        return None, "No match found at all — check that both sequences are the same gene."
    # Greedy extension can run a few bases past the true exon end whenever the start of the
    # intron happens to match the start of the next exon. The junction can be slid left by t
    # bases when the moved bases are identical in both places; pick the slide that makes the
    # intron canonical (starts GT, ends AG).
    for i in range(len(exons) - 1):
        b1, a2 = exons[i][1], exons[i + 1][0]
        for t in range(0, 6):
            if t and g[b1 - t:b1] != g[a2 - t:a2]:
                break
            nb1, na2 = b1 - t, a2 - t
            if nb1 <= exons[i][0] or na2 >= exons[i + 1][1]:
                break
            if g[nb1:nb1 + 2] == "GT" and g[na2 - 2:na2] == "AG":
                exons[i] = (exons[i][0], nb1)
                exons[i + 1] = (na2, exons[i + 1][1])
                break
    return [(s + 1, e) for s, e in exons], None


def splice_sites(g, exons):
    """Canonical introns start GT and end AG. Returns one row per intron."""
    out = []
    for i in range(len(exons) - 1):
        a, b = exons[i][1], exons[i + 1][0] - 1
        don, acc = g[a:a + 2], g[b - 2:b]
        out.append({"intron": i + 1, "from": a + 1, "to": b, "length": b - a,
                    "donor": don, "acceptor": acc,
                    "canonical GT–AG": "yes" if (don == "GT" and acc == "AG") else "no"})
    return out


def downstream_atgs(cds, cut_in_cds, max_scan_aa=200, limit=3):
    hits, start = [], max(cut_in_cds, 0)
    start += (-start) % 3
    for i in range(start, min(len(cds) - 2, start + max_scan_aa * 3), 3):
        if cds[i:i + 3] == "ATG":
            prot = translate(cds[i:])
            stop = prot.find("*")
            hits.append({"nt_pos": i + 1, "aa_pos": i // 3 + 1,
                         "product_aa": stop if stop != -1 else len(prot)})
            if len(hits) >= limit:
                break
    return hits


def mmej_inframe(seq, cut, window=20):
    left, right = seq[max(0, cut - window):cut], seq[cut:cut + window]
    if len(left) < 6 or len(right) < 6:
        return None, []
    pairs = []
    for n in range(3, 9):
        for i in range(len(left) - n + 1):
            motif = left[i:i + n]
            for j in range(len(right) - n + 1):
                if right[j:j + n] == motif:
                    d = (len(left) - i) + j
                    if 1 <= d <= 30:
                        pairs.append({"microhomology": motif, "length": n,
                                      "deletion (bp)": d, "in frame": d % 3 == 0})
    if not pairs:
        return None, []
    w = sum(2 ** p["length"] for p in pairs)
    wi = sum(2 ** p["length"] for p in pairs if p["in frame"])
    return 100 * wi / w, sorted(pairs, key=lambda p: (-p["length"], p["deletion (bp)"]))[:12]


def score_guide(cds_pct, sc, n_match, atg_risk, gc_val, spacer):
    pts, notes = 0, []
    if 5 <= cds_pct <= 30:
        pts += 25
    elif 30 < cds_pct <= 50:
        pts += 15
        notes.append("mid-CDS (later than ideal)")
    elif cds_pct < 5:
        pts += 8
        notes.append("very close to ATG — reinitiation risk")
    else:
        notes.append("late in CDS — truncated protein may stay functional")
    if 40 <= gc_val <= 60:
        pts += 20
    elif 30 <= gc_val <= 70:
        pts += 12
        notes.append(f"GC {gc_val:.0f}%")
    else:
        notes.append(f"GC {gc_val:.0f}% — outside 30–70%")
    if "TTTT" in spacer:
        notes.append("TTTT — may terminate U6 transcription")
    else:
        pts += 15
    if re.search(r"(A{5,}|C{5,}|G{5,}|T{5,})", spacer):
        notes.append("homopolymer run ≥5")
    else:
        pts += 5
    if len(sc) >= 6:
        notes.append(f"self-complementary ({sc})")
    else:
        pts += 10
    if n_match > 1:
        notes.append(f"{n_match} exact matches in input")
    else:
        pts += 10
    if atg_risk:
        notes.append("in-frame ATG close after cut — reinitiation risk")
    else:
        pts += 15
    return pts, notes


# ------------------------------------------------------------------ the sequence map
# Colours: validated categorical slots 1-3 (blue / orange / aqua), light + dark steps.
# Cut markers and all text use neutral ink tokens so hue only ever carries strand identity.
MAP_CSS = """
.viz-root{color-scheme:light;
 --surface-1:#fcfcfb; --surface-2:#f0efec; --surface-3:#e3e2dd;
 --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#78776f;
 --plus:#2a78d6; --plus-deep:#184f95; --minus:#eb6834; --minus-deep:#9c3d16; --span:#1baf7a;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{
 color-scheme:dark;
 --surface-1:#1a1a19; --surface-2:#262625; --surface-3:#343432;
 --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#96958c;
 --plus:#3987e5; --plus-deep:#9ec5f4; --minus:#d95926; --minus-deep:#f0a681; --span:#199e70;}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;
 --surface-1:#1a1a19; --surface-2:#262625; --surface-3:#343432;
 --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#96958c;
 --plus:#3987e5; --plus-deep:#9ec5f4; --minus:#d95926; --minus-deep:#f0a681; --span:#199e70;}
html,body{margin:0;padding:0;background:transparent;}
.viz-root{background:var(--surface-1);border-radius:10px;padding:2px 0;
 font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;}
.viz-root svg{display:block;}
text{font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;}
.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;}
.lbl{font-size:11px;fill:var(--text-primary);}
.sub{font-size:10px;fill:var(--text-secondary);}
.mut{font-size:9.5px;fill:var(--text-muted);}
.gbar{cursor:default;}
.gbar:hover rect.body{stroke:var(--text-primary);stroke-width:1.2;}
"""


def svg_map(seq, cds_lo, cds_hi, exons, sel, view_lo, view_hi, theme, labels=None, map_w=1120):
    """sel: list of guide dicts (already the ones to draw). Returns (html document, height)."""
    W, ML, MR = float(map_w), 54.0, 34.0
    PW = W - ML - MR
    span_bp = max(view_hi - view_lo, 1)
    # draw individual bases whenever each one gets enough room to be legible, so a wider map
    # shows bases across a wider window
    show_bases = PW / span_bp >= 5.5
    labels = labels or {}

    def x(pos):
        return ML + (pos - view_lo) / span_bp * PW

    plus = [g for g in sel if g["strand"] == "+"]
    minus = [g for g in sel if g["strand"] == "-"]

    def pack(gs):
        """greedy row assignment so bars + labels don't overlap"""
        rows = []
        for g in sorted(gs, key=lambda d: d["sp_lo"]):
            x0 = x(min(g["sp_lo"], g["pam_lo"]))
            lbl = labels.get(id(g), g["spacer"][:8])
            x1 = max(x(max(g["sp_hi"], g["pam_hi"])), x0 + 8 * len(str(lbl)) + 46)
            for r in rows:
                if x0 > r[-1][1] + 10:
                    r.append((x0, x1, g))
                    break
            else:
                rows.append([(x0, x1, g)])
        return rows

    rp, rm = pack(plus), pack(minus)
    two = len(sel) == 2
    cuts = sorted(g["cut"] for g in sel)

    ROW, BAR = 30.0, 15.0
    y = 14.0
    y_span = None
    if two:
        y_span = y + 11
        y += 38
    y_plus_top = y
    y += len(rp) * ROW + (8 if rp else 0)
    y_band, BH = y, 28.0
    y += BH
    y_bases = None
    if show_bases:
        y_bases = y + 15
        y += 21
    y_ruler = y + 14
    y += 26
    y_minus_top = y
    y += len(rm) * ROW + (6 if rm else 0)
    y_leg = y + 12
    H = y + 32

    p = []
    a = p.append
    a(f'<svg viewBox="0 0 {W:.0f} {H:.0f}" width="{W:.0f}" height="{H:.0f}" role="img" '
      f'aria-label="Sequence map of {html.escape(str(len(seq)))} bp with {len(sel)} guides">')
    a('<defs>'
      '<marker id="ar-p" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" '
      'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--plus)"/></marker>'
      '<marker id="ar-m" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" '
      'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--minus)"/></marker>'
      '</defs>')

    # ---- backbone band
    a(f'<rect x="{ML:.1f}" y="{y_band:.1f}" width="{PW:.1f}" height="{BH:.1f}" rx="5" '
      f'fill="var(--surface-2)" stroke="var(--surface-3)"/>')
    # CDS
    if cds_hi > cds_lo:
        cl, ch = max(cds_lo, view_lo), min(cds_hi, view_hi)
        if ch > cl:
            a(f'<rect x="{x(cl):.1f}" y="{y_band + 4:.1f}" width="{max(x(ch) - x(cl), 1):.1f}" '
              f'height="{BH - 8:.1f}" rx="3" fill="var(--surface-3)"/>'
              f'<title>CDS {cds_lo + 1}–{cds_hi}</title>')
            if x(ch) - x(cl) > 34:
                a(f'<text class="mut" x="{(x(cl) + x(ch)) / 2:.1f}" y="{y_band + BH / 2 + 3.5:.1f}" '
                  f'text-anchor="middle">CDS</text>')
    # exons
    for i, (es, ee) in enumerate(exons, 1):
        el, eh = max(es - 1, view_lo), min(ee, view_hi)
        if eh > el:
            a(f'<g class="gbar"><rect x="{x(el):.1f}" y="{y_band + 1.5:.1f}" '
              f'width="{max(x(eh) - x(el), 1.5):.1f}" height="{BH - 3:.1f}" rx="3" '
              f'fill="none" stroke="var(--text-muted)" stroke-width="1.2" stroke-dasharray="3 2"/>'
              f'<title>exon {i}: {es}–{ee} ({ee - es + 1} bp)</title></g>')

    # ---- bases (zoomed in)
    if show_bases:
        step = PW / span_bp
        hi_sp = {}
        for g in sel:
            for q in range(g["sp_lo"], g["sp_hi"]):
                hi_sp[q] = "sp-" + g["strand"]
            for q in range(g["pam_lo"], g["pam_hi"]):
                hi_sp[q] = "pam-" + g["strand"]
        fs = min(12.5, max(6.0, step * 1.25))
        for q in range(view_lo, view_hi):
            if q >= len(seq):
                break
            kind = hi_sp.get(q)
            col = "var(--text-secondary)"
            weight = "400"
            if kind == "sp-+":
                col, weight = "var(--plus)", "700"
            elif kind == "sp--":
                col, weight = "var(--minus)", "700"
            elif kind == "pam-+":
                col, weight = "var(--plus-deep)", "700"
            elif kind == "pam--":
                col, weight = "var(--minus-deep)", "700"
            a(f'<text class="mono" x="{x(q) + step / 2:.1f}" y="{y_bases:.1f}" '
              f'text-anchor="middle" font-size="{fs:.1f}" font-weight="{weight}" '
              f'fill="{col}">{seq[q]}</text>')

    # ---- ruler
    n_ticks = 9
    for i in range(n_ticks + 1):
        pos = view_lo + span_bp * i / n_ticks
        a(f'<line x1="{x(pos):.1f}" y1="{y_ruler - 11:.1f}" x2="{x(pos):.1f}" '
          f'y2="{y_ruler - 6:.1f}" stroke="var(--text-muted)" stroke-width="1"/>')
        a(f'<text class="mut" x="{x(pos):.1f}" y="{y_ruler + 3:.1f}" text-anchor="middle">'
          f'{int(pos) + 1:,}</text>')
    a(f'<line x1="{ML:.1f}" y1="{y_ruler - 11:.1f}" x2="{ML + PW:.1f}" y2="{y_ruler - 11:.1f}" '
      f'stroke="var(--surface-3)" stroke-width="1"/>')

    # ---- guides
    def draw(gs_rows, strand, top):
        up = strand == "+"
        hue = "var(--plus)" if up else "var(--minus)"
        deep = "var(--plus-deep)" if up else "var(--minus-deep)"
        for ri, row in enumerate(gs_rows):
            # rows stack away from the band
            if up:
                by = top + (len(gs_rows) - 1 - ri) * ROW
            else:
                by = top + ri * ROW
            for _, _, g in row:
                sx0, sx1 = x(g["sp_lo"]), x(g["sp_hi"])
                px0, px1 = x(g["pam_lo"]), x(g["pam_hi"])
                w = max(sx1 - sx0, 2.5)
                lbl = str(labels.get(id(g), ""))
                a('<g class="gbar">')
                a(f'<title>{html.escape(lbl + ("  " if lbl else ""))}'
                  f'{g["spacer"]} + PAM {g["pam"]} &#183; {strand} strand &#183; '
                  f'spacer {g["sp_lo"] + 1}–{g["sp_hi"]} &#183; cut after base {g["cut"]}</title>')
                # spacer bar, 4px rounded ends, arrowhead in the strand direction
                a(f'<rect class="body" x="{sx0:.1f}" y="{by:.1f}" width="{w:.1f}" '
                  f'height="{BAR:.1f}" rx="4" fill="{hue}"/>')
                ay = by + BAR / 2
                if up:
                    a(f'<line x1="{sx1:.1f}" y1="{ay:.1f}" x2="{sx1 + 7:.1f}" y2="{ay:.1f}" '
                      f'stroke="{hue}" stroke-width="2.5" marker-end="url(#ar-p)"/>')
                else:
                    a(f'<line x1="{sx0:.1f}" y1="{ay:.1f}" x2="{sx0 - 7:.1f}" y2="{ay:.1f}" '
                      f'stroke="{hue}" stroke-width="2.5" marker-end="url(#ar-m)"/>')
                # PAM, separated from the spacer by a 2px surface gap
                a(f'<rect x="{min(px0, px1) + 1:.1f}" y="{by + 2.5:.1f}" '
                  f'width="{max(abs(px1 - px0) - 2, 1.5):.1f}" height="{BAR - 5:.1f}" rx="2" '
                  f'fill="{deep}"/>')
                # dashed drop line to the cut on the band + ink cut marker
                cx = x(g["cut"])
                y1 = by + BAR if up else y_band + BH
                y2 = y_band if up else by
                a(f'<line x1="{cx:.1f}" y1="{y1:.1f}" x2="{cx:.1f}" y2="{y2:.1f}" '
                  f'stroke="var(--text-muted)" stroke-width="1" stroke-dasharray="2 2.5"/>')
                cut_bot = (y_bases + 4) if y_bases is not None else (y_band + BH + 4)
                a(f'<line x1="{cx:.1f}" y1="{y_band - 4:.1f}" x2="{cx:.1f}" '
                  f'y2="{cut_bot:.1f}" stroke="var(--text-primary)" stroke-width="1.6"/>')
                a(f'<circle cx="{cx:.1f}" cy="{y_band + (-7 if up else BH + 7):.1f}" r="2.6" '
                  f'fill="var(--text-primary)"/>')
                # direct label in ink, with a hue swatch carrying identity
                tx = min(max(sx1, px1) + 10, W - MR - 6)
                anchor = "start"
                if tx > W - MR - 120:
                    tx, anchor = min(sx0, px0) - 10, "end"
                a(f'<rect x="{tx - (7 if anchor == "start" else -1):.1f}" y="{by + 3.5:.1f}" '
                  f'width="0" height="0" fill="none"/>')
                a(f'<text class="lbl" x="{tx:.1f}" y="{by + BAR / 2 + 4:.1f}" '
                  f'text-anchor="{anchor}">{html.escape(lbl)}'
                  f'<tspan class="sub"> cut {g["cut"]:,}</tspan></text>')
                a('</g>')

    draw(rp, "+", y_plus_top)
    draw(rm, "-", y_minus_top)

    # ---- deletion span between two cuts
    if two and y_span is not None:
        c0, c1 = cuts[0], cuts[1]
        xa, xb = x(c0), x(c1)
        a(f'<line x1="{xa:.1f}" y1="{y_span:.1f}" x2="{xb:.1f}" y2="{y_span:.1f}" '
          f'stroke="var(--span)" stroke-width="3" stroke-linecap="round"/>')
        for xx in (xa, xb):
            a(f'<line x1="{xx:.1f}" y1="{y_span - 5:.1f}" x2="{xx:.1f}" y2="{y_span + 5:.1f}" '
              f'stroke="var(--span)" stroke-width="2.5" stroke-linecap="round"/>')
        d = c1 - c0
        a(f'<text class="lbl" x="{(xa + xb) / 2:.1f}" y="{y_span - 9:.1f}" text-anchor="middle" '
          f'font-weight="600">&#916; {d:,} bp'
          f'<tspan class="sub" dx="10">{"in frame" if d % 3 == 0 else "frameshift"}</tspan></text>')

    # ---- legend (always present for >=2 series); the guide table is the table view
    items = [("sw", "var(--plus)", "+ strand guide"), ("sw", "var(--minus)", "− strand guide")]
    if two:
        items.append(("sw", "var(--span)", "deleted region"))
    items += [("sw", "var(--plus-deep)", "PAM"), ("ln", "var(--text-primary)", "cut site")]
    col_w = PW / len(items)
    for i, (kind, col, txt) in enumerate(items):
        lx = ML + i * col_w
        if kind == "sw":
            a(f'<rect x="{lx:.1f}" y="{y_leg - 8:.1f}" width="11" height="11" rx="3" fill="{col}"/>')
        else:
            a(f'<line x1="{lx + 5:.1f}" y1="{y_leg - 9:.1f}" x2="{lx + 5:.1f}" '
              f'y2="{y_leg + 3:.1f}" stroke="{col}" stroke-width="1.8"/>')
        a(f'<text class="sub" x="{lx + 17:.1f}" y="{y_leg + 1:.1f}">{txt}</text>')
    a('</svg>')

    attr = f' data-theme="{theme}"' if theme in ("light", "dark") else ""
    return (f'<!DOCTYPE html><html{attr}><head><meta charset="utf-8">'
            f'<style>{MAP_CSS}</style></head><body>'
            f'<div class="viz-root">{"".join(p)}</div></body></html>'), H


# ------------------------------------------------------------------ UI
st.title("✂️ Mammalian KO Designer")
st.caption("Pick where to cut, see the guides on your sequence, get the oligos.")

with st.expander("⚠️ What this app does and does not do", expanded=False):
    st.markdown("""
**It does:** find guides, rank them by published *design rules*, place each cut in the coding
sequence, flag downstream in-frame ATGs that could rescue a truncated protein, estimate whether
microhomology repair tends to be in-frame, draw it all on a map, and write your cloning oligos.

**It does not:** search the genome for off-targets, or run a trained on-target model
(Doench Rule Set 2, DeepSpCas9…). Those need the whole genome or a model file.

**So:** use this to pick *where* and *how* to cut and to get your oligos, then paste your final
2–3 spacers into **CRISPOR** or **CHOPCHOP** for the off-target check before ordering.
Also confirm in Ensembl that your target exon is in **all** transcript isoforms — this app
cannot know that from a single pasted sequence.
""")

st.sidebar.header("Input")
mode = st.sidebar.radio("I will paste", ["Coding sequence (starts at ATG)",
                                         "Genomic / larger region (I'll mark the CDS)"])
nuclease = st.sidebar.selectbox("Nuclease", ["SpCas9 (NGG)", "SaCas9 (NNGRRT)",
                                             "Cas12a / Cpf1 (TTTV)"])
spacer_len = st.sidebar.slider("Spacer length (Cas9)", 17, 24, 20)
win_lo, win_hi = st.sidebar.slider("Preferred CDS window (%)", 0, 100, (5, 30))
st.sidebar.caption("Early coding exons are preferred: an early frameshift is more likely to "
                   "destroy the protein. Avoid cutting before ~5% — reinitiation at a nearby "
                   "downstream ATG can rescue it.")
map_theme = st.sidebar.radio("Map colours", ["Auto", "Light", "Dark"], horizontal=True)
theme = map_theme.lower()
map_w = st.sidebar.slider("Map width (px)", 700, 2400, 1120, step=20)
st.sidebar.caption("The map is drawn at exactly this width. If it is cut off, lower this or "
                   "widen the preview pane; raise it on a big screen. The map scrolls sideways "
                   "if it does not fit.")

seq_text = st.text_area("Paste your sequence (FASTA or plain)", height=140,
                        placeholder=">MyGene CDS\nATGGCCGAGCTGTCT...")
name, seq = parse_fasta_one(seq_text)

if not seq:
    st.info("Paste a sequence to start. The coding sequence starting at ATG is enough for a "
            "knockout. Paste genomic sequence if you want exon context on the map.")
    st.stop()

if mode.startswith("Coding"):
    cds_lo, cds_hi = 0, len(seq) - len(seq) % 3
    if not seq.startswith("ATG"):
        st.warning("Your sequence does not start with ATG. Check you pasted the CDS, or switch "
                   "to the genomic option and mark the CDS.")
else:
    c1, c2 = st.columns(2)
    cds_lo = c1.number_input("CDS starts at base (1-based)", 1, len(seq), 1) - 1
    cds_hi = c2.number_input("CDS ends at base (1-based)", 1, len(seq), len(seq))
cds = seq[cds_lo:cds_hi]
prot = translate(cds)
full_aa = max(len(prot.split("*")[0]), 1)

with st.expander("Exons — find them automatically, or type them in", expanded=False):
    st.markdown("""
Exon boundaries cannot be read out of a genomic sequence on its own — splice sites are just
`GT`…`AG` pairs that occur by chance all over the place. But the **mRNA (or CDS) is the exons
with the introns removed**, so aligning it back onto the genomic sequence recovers the
boundaries exactly.

Paste the genomic sequence in the main box above, and the matching mRNA or CDS here. From
Ensembl, that is the same transcript with `type=genomic` and `type=cds`.
""")
    mrna_text = st.text_area("mRNA or CDS of the same transcript (FASTA or plain)", height=110,
                             key="mrna_in", placeholder=">ENST… cds\nATGGCC...")
    if st.button("Find exons", key="findex"):
        _, mrna = parse_fasta_one(mrna_text)
        if not mrna:
            st.warning("Paste the mRNA or CDS first.")
        else:
            found, err = spliced_align(seq, mrna)
            if err:
                st.error(err)
                st.caption("If the gene is on the minus strand, reverse-complement one of them "
                           "and try again — or fetch both from Ensembl, which returns them in "
                           "matching orientation.")
            else:
                st.session_state["found_exons"] = found
                st.session_state["exon_text"] = ", ".join(f"{s}-{e}" for s, e in found)
                st.success(f"Found {len(found)} exons covering {sum(e - s + 1 for s, e in found):,} bp "
                           f"of your {len(seq):,} bp sequence.")
    if st.session_state.get("found_exons"):
        fe = st.session_state["found_exons"]
        st.dataframe(pd.DataFrame([{"exon": i, "start": s, "end": e, "length": e - s + 1}
                                   for i, (s, e) in enumerate(fe, 1)]), hide_index=True)
        ss = splice_sites(seq, fe)
        if ss:
            bad = [r for r in ss if r["canonical GT–AG"] == "no"]
            (st.warning if bad else st.success)(
                f"{len(ss) - len(bad)} of {len(ss)} introns start GT and end AG."
                + (" The others may be non-canonical, or that boundary may be misplaced — "
                   "check those rows." if bad else " That is the expected pattern, so the "
                                                   "boundaries look right."))
            st.dataframe(pd.DataFrame(ss), hide_index=True)
        if st.button("Clear", key="clrex"):
            st.session_state.pop("found_exons", None)
            st.session_state["exon_text"] = ""
            st.rerun()

exon_text = st.text_input("Exon boundaries as start-end pairs (filled in automatically if you "
                          "used the finder above; edit freely)", key="exon_text")
exons = [(int(q.group(1)), int(q.group(2)))
         for q in (re.match(r"\s*(\d+)\s*-\s*(\d+)\s*$", t) for t in exon_text.split(",")) if q]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Sequence", f"{len(seq):,} bp")
m2.metric("CDS", f"{len(cds):,} bp")
m3.metric("Protein", f"{full_aa:,} aa")
m4.metric("CDS GC", f"{gc(cds):.1f}%")
if "*" in prot[:-1]:
    st.warning("There is an internal stop codon in the CDS — check the frame and coordinates.")

# ---- build the guide table once
all_guides = find_guides(seq, nuclease, spacer_len)
rows = []
for g in all_guides:
    if not (cds_lo <= g["cut"] <= cds_hi):
        continue
    cic = g["cut"] - cds_lo
    pct = 100 * cic / max(len(cds), 1)
    sp = g["spacer"]
    gcv, scomp = gc(sp), max_self_comp(sp)
    nm = len(re.findall(f"(?={re.escape(sp)})", seq)) + len(re.findall(f"(?={re.escape(rc(sp))})", seq))
    atgs = downstream_atgs(cds, cic, 60)
    risk = bool(atgs and atgs[0]["aa_pos"] - cic // 3 <= 40 and atgs[0]["product_aa"] > 0.5 * full_aa)
    pts, notes = score_guide(pct, scomp, nm, risk, gcv, sp)
    mh, _ = mmej_inframe(seq, g["cut"])
    exon_no = ""
    for i, (es, ee) in enumerate(exons, 1):
        if es <= g["cut"] + 1 <= ee:
            exon_no = str(i)
            if min(g["cut"] + 1 - es, ee - (g["cut"] + 1)) < 10:
                notes.append("close to a splice site")
    rows.append({"show": False, "score": pts, "strand": g["strand"], "spacer": sp,
                 "PAM": g["pam"], "cut": g["cut"], "codon": cic // 3 + 1,
                 "CDS %": round(pct, 1), "exon": exon_no, "GC %": round(gcv, 1),
                 "5'G": "yes" if sp.startswith("G") else "add",
                 "in-frame MMEJ %": "—" if mh is None else str(round(mh)),
                 "next ATG codon": str(atgs[0]["aa_pos"]) if atgs else "—",
                 "flags": "; ".join(notes) or "✓ clean", "_g": g})
rows.sort(key=lambda r: -r["score"])

tabs = st.tabs(["1 · Strategy", "2 · Guides & map", "3 · Two-guide deletion", "4 · Cloning oligos"])

# ---------------------------------------------------------------- 1. strategy
with tabs[0]:
    st.subheader("What is the best way to knock this out?")
    wn = (int(len(cds) * win_lo / 100), int(len(cds) * win_hi / 100))
    early = [a for a in downstream_atgs(cds, 0, 200, limit=6) if 1 < a["aa_pos"] <= 100]
    st.markdown(f"""
**Target:** {name} — {len(cds):,} bp CDS, {full_aa:,} aa protein.

**Recommended approach — frameshift by a single cut (indel KO).** The standard route in
mammalian cells: one guide, end-joining makes a small insertion or deletion, and a 1 or 2 bp
shift destroys the reading frame. Aim the cut between **base {wn[0]} and {wn[1]}** of the CDS
(codon {wn[0] // 3 + 1}–{max(wn[1] // 3, 1)}), your {win_lo}–{win_hi}% window.
""")
    if early:
        st.warning(f"""**Reinitiation risk.** In-frame ATGs sit early in this CDS (codon
{', '.join(str(a['aa_pos']) for a in early[:4])}). If you cut *before* one, translation can
restart there and make a shortened protein that may still work — a known cause of knockouts
that edit well but keep the function. Prefer a cut **after** codon
{max(a['aa_pos'] for a in early[:4])}, or delete the region containing them (tab 3).""")
    else:
        st.success("No in-frame ATG in the first 100 codons — low reinitiation risk for an early cut.")
    st.markdown("""
**When to use two guides instead (a real deletion):**
- a secondary start site you cannot avoid
- you want to remove one domain and keep the rest
- the gene is short, or you want a deletion you can see on a gel
- a non-coding gene (miRNA, lncRNA), where a frameshift means nothing

**Checks this app cannot do for you**
1. Is your target exon in **every** transcript isoform? Check Ensembl.
2. Is it the **last** exon? A frameshift there often escapes nonsense-mediated decay.
3. **Off-targets** — run your final spacers through CRISPOR or CHOPCHOP.
4. Is the gene **essential**? Then expect to need an inducible or conditional approach.
""")
    if exons:
        st.markdown("**Your exons, relative to the CDS**")
        st.dataframe(pd.DataFrame([{
            "exon": i, "start": s, "end": e, "length": e - s + 1,
            "CDS % at midpoint": round(((s + e) / 2 - cds_lo) / max(len(cds), 1) * 100, 1),
            "in preferred window": "yes" if win_lo <= ((s + e) / 2 - cds_lo) / max(len(cds), 1) * 100 <= win_hi else "no",
            "length ÷ 3 remainder": (e - s + 1) % 3} for i, (s, e) in enumerate(exons, 1)]),
            hide_index=True)
        st.caption("An exon whose length is not a multiple of 3 shifts the frame if it is skipped — "
                   "that makes it a good single-cut target.")

# ---------------------------------------------------------------- 2. guides & map
with tabs[1]:
    if not rows:
        st.warning("No guides found inside the CDS. Check your coordinates, or try another nuclease.")
    else:
        only_win = st.checkbox("Only guides cutting inside the preferred CDS window", True)
        view = [r for r in rows if (not only_win) or (win_lo <= r["CDS %"] <= win_hi)]
        if not view:
            st.warning("No guides in that window — widen it in the sidebar, or untick the box.")
            view = rows
        st.caption(f"{len(view)} guides. **Tick `show`** to put a guide on the map below.")

        df = pd.DataFrame([{k: v for k, v in r.items() if k != "_g"} for r in view])
        edited = st.data_editor(
            df, hide_index=True, height=330, key="guide_table",
            disabled=[c for c in df.columns if c != "show"],
            column_config={
                "show": st.column_config.CheckboxColumn("show", help="Draw this guide on the map",
                                                        default=False, width="small"),
                "score": st.column_config.NumberColumn("score", help="rule-based, 0–100"),
                "cut": st.column_config.NumberColumn("cut", help="cut after this base"),
            })
        picked = [view[i]["_g"] for i in range(len(view)) if bool(edited.iloc[i]["show"])]
        lbl = {id(g): f"g{i + 1}" for i, g in enumerate(sorted(picked, key=lambda d: d["cut"]))}

        st.divider()
        st.markdown("### Sequence map")
        if not picked:
            st.info("Tick `show` on one or more guides above — they will be drawn here, on the "
                    "strand they bind, with the cut site marked.")
            vlo, vhi = 0, len(seq)
        else:
            zoom = st.radio("View", ["Whole sequence", "Zoom to selected guides",
                                     "Nucleotides at each cut", "Custom window"],
                            horizontal=True, key="zoomsel")
            if zoom == "Whole sequence":
                vlo, vhi = 0, len(seq)
            elif zoom == "Zoom to selected guides":
                pad = st.slider("Padding around the selection (bp)", 10, 500, 60, key="zpad")
                lo = min(min(g["sp_lo"], g["pam_lo"]) for g in picked)
                hi = max(max(g["sp_hi"], g["pam_hi"]) for g in picked)
                vlo, vhi = max(0, lo - pad), min(len(seq), hi + pad)
            elif zoom == "Custom window":
                vlo, vhi = st.slider("Window (base)", 1, len(seq),
                                     (1, min(len(seq), 400)), key="cwin")
                vlo -= 1
            else:
                vlo, vhi = None, None
            if vlo is not None and vhi - vlo < 20:
                vhi = min(len(seq), vlo + 20)

        if picked and vlo is None:
            # one panel per cut, each narrow enough that individual bases are drawn
            half = st.slider("Bases shown either side of each cut", 20, 90, 45, key="ncut")
            for g in sorted(picked, key=lambda d: d["cut"]):
                st.markdown(f"**{lbl[id(g)]}** · `{g['spacer']}` + PAM `{g['pam']}` · "
                            f"{g['strand']} strand · cut after base {g['cut']:,}")
                a_, b_ = max(0, g["cut"] - half), min(len(seq), g["cut"] + half)
                doc, h = svg_map(seq, cds_lo, cds_hi, exons, [g], a_, b_, theme, lbl, map_w)
                components.html(doc, height=int(h) + 26, scrolling=True)
        else:
            doc, h = svg_map(seq, cds_lo, cds_hi, exons, picked, vlo, vhi, theme, lbl, map_w)
            components.html(doc, height=int(h) + 26, scrolling=True)
        if picked:
            st.caption("Hover any bar for its spacer, PAM, strand and cut position. Individual "
                       "bases appear once each one has room — zoom in, widen the map, or use "
                       "“Nucleotides at each cut” when your guides are far apart.")

        # per-guide detail
        if picked:
            st.divider()
            st.markdown("### Guide detail")
            which = st.selectbox("Guide", [f'{lbl[id(g)]} · {g["spacer"]} ({g["strand"]})'
                                           for g in sorted(picked, key=lambda d: d["cut"])])
            g = [q for q in picked if lbl[id(q)] == which.split(" · ")[0]][0]
            cut, cic = g["cut"], g["cut"] - cds_lo
            c1, c2 = st.columns([3, 2])
            with c1:
                lo2, hi2 = max(0, cut - 30), min(len(seq), cut + 30)
                st.code(f"{seq[lo2:cut]} | {seq[cut:hi2]}\n{' ' * (cut - lo2)}^ cut after base {cut:,}",
                        language=None)
                st.write(f"Protospacer binds bases **{g['sp_lo'] + 1:,}–{g['sp_hi']:,}** on the "
                         f"**{g['strand']} strand**; PAM `{g['pam']}` at "
                         f"**{g['pam_lo'] + 1:,}–{g['pam_hi']:,}**.")
                mh, pairs = mmej_inframe(seq, cut)
                if pairs:
                    st.markdown(f"**Microhomology at this cut** — roughly **{mh:.0f}%** of "
                                "microhomology-driven deletions here would be in frame "
                                "(lower is better for a knockout).")
                    st.dataframe(pd.DataFrame(pairs), hide_index=True)
            with c2:
                st.markdown("**If the frame shifts here**")
                st.write(f"Protein made before the cut: **{max(cic // 3, 0):,} aa** "
                         f"({100 * max(cic // 3, 0) / full_aa:.0f}% of full length)")
                at = downstream_atgs(cds, cic, 120)
                if at:
                    st.markdown("**Downstream in-frame ATGs**")
                    st.dataframe(pd.DataFrame(at).rename(columns={
                        "nt_pos": "CDS base", "aa_pos": "codon",
                        "product_aa": "protein it could make (aa)"}), hide_index=True)
                    st.caption("A sizeable product here means reinitiation could rescue the gene.")
                else:
                    st.success("No in-frame ATG shortly after the cut.")

# ---------------------------------------------------------------- 3. two guides
with tabs[2]:
    st.subheader("Delete a region with two guides")
    if len(rows) < 2:
        st.warning("Need at least two guides in the CDS.")
    else:
        st.caption("Pick the guide that cuts on the left and the one that cuts on the right. "
                   "The map shows the piece that goes.")
        opts = sorted(rows, key=lambda r: r["cut"])
        names = [f'{r["cut"]:,} · {r["spacer"]} ({r["strand"]}, score {r["score"]})' for r in opts]
        c1, c2 = st.columns(2)
        i_l = c1.selectbox("Left guide (cuts first)", range(len(opts)),
                           format_func=lambda i: names[i], index=0)
        i_r = c2.selectbox("Right guide (cuts second)", range(len(opts)),
                           format_func=lambda i: names[i], index=len(opts) - 1)
        gl, gr = opts[i_l], opts[i_r]
        if gl["cut"] >= gr["cut"]:
            st.error("The left guide must cut before the right one. Swap them.")
        else:
            d = gr["cut"] - gl["cut"]
            pair = [gl["_g"], gr["_g"]]
            plbl = {id(gl["_g"]): "L", id(gr["_g"]): "R"}
            k1, k2, k3 = st.columns(3)
            k1.metric("Deletion", f"{d:,} bp")
            k2.metric("Frame", "in frame" if d % 3 == 0 else "frameshift")
            k3.metric("Protein lost", f"{d // 3:,} aa")
            if d % 3 == 0:
                st.warning("This deletion is a multiple of 3, so the rest of the protein stays in "
                           "frame — an internally deleted protein may still fold and work. For a "
                           "clean knockout prefer a deletion that is **not** a multiple of 3, or "
                           "one that removes the start codon.")
            else:
                st.success("Not a multiple of 3 — this shifts the frame as well as deleting, "
                           "which is what you want for a knockout.")
            zoom2 = st.radio("View", ["Zoom to the deletion", "Whole sequence",
                                      "Nucleotides at each cut"], horizontal=True, key="z2")
            if zoom2 == "Nucleotides at each cut":
                half2 = st.slider("Bases shown either side of each cut", 20, 90, 45, key="ncut2")
                st.caption("The two cuts are far apart, so they are shown as two close-ups. "
                           "Everything between them is deleted.")
                for side, g in (("L", gl), ("R", gr)):
                    st.markdown(f"**{side}** · `{g['spacer']}` + PAM `{g['PAM']}` · "
                                f"{g['strand']} strand · cut after base {g['cut']:,}")
                    a_, b_ = max(0, g["cut"] - half2), min(len(seq), g["cut"] + half2)
                    doc, h = svg_map(seq, cds_lo, cds_hi, exons, [g["_g"]], a_, b_, theme,
                                     plbl, map_w)
                    components.html(doc, height=int(h) + 26, scrolling=True)
            else:
                if zoom2 == "Whole sequence":
                    v2 = (0, len(seq))
                else:
                    pad = max(40, d // 8)
                    v2 = (max(0, gl["cut"] - pad), min(len(seq), gr["cut"] + pad))
                doc, h = svg_map(seq, cds_lo, cds_hi, exons, pair, v2[0], v2[1], theme, plbl, map_w)
                components.html(doc, height=int(h) + 26, scrolling=True)

            st.markdown("**Both spacers**")
            st.dataframe(pd.DataFrame([
                {"side": "L", "spacer": gl["spacer"], "PAM": gl["PAM"], "strand": gl["strand"],
                 "cut": gl["cut"], "GC %": gl["GC %"], "score": gl["score"], "flags": gl["flags"]},
                {"side": "R", "spacer": gr["spacer"], "PAM": gr["PAM"], "strand": gr["strand"],
                 "cut": gr["cut"], "GC %": gr["GC %"], "score": gr["score"], "flags": gr["flags"]},
            ]), hide_index=True)

            st.divider()
            st.subheader("Optional HDR template for a clean deletion")
            arm = st.slider("Homology arm length (bp)", 30, 1000, 500, step=10)
            la, ra = seq[max(0, gl["cut"] - arm):gl["cut"]], seq[gr["cut"]:gr["cut"] + arm]
            if len(la) < arm or len(ra) < arm:
                st.info(f"Your pasted sequence only supports arms of {len(la)} bp (left) and "
                        f"{len(ra)} bp (right). Paste more flanking sequence for longer arms.")
            st.code(f">{name}_HDR_del_{gl['cut']}_{gr['cut']}\n{la}{ra}", language=None)
            st.caption("~40–60 bp arms suit an ssODN for small edits; ~500–1000 bp for a plasmid "
                       "or a large deletion. Mutate the PAM in the template so it is not re-cut.")

# ---------------------------------------------------------------- 4. oligos
with tabs[3]:
    st.subheader("Cloning oligos")
    vec = st.selectbox("Vector / backbone", list(VECTORS))
    enzyme, fwd_oh, rev_oh = VECTORS[vec]
    choices = ["(type my own)"] + [r["spacer"] for r in rows[:40]]
    pre = st.selectbox("Spacer", choices)
    sp_in = st.text_input("Spacer (no PAM)", "" if pre == "(type my own)" else pre)
    sp_in = re.sub(r"[^ACGT]", "", sp_in.strip().upper())
    if sp_in:
        add_g = not sp_in.startswith("G")
        # sense:     CACC [G if needed] spacer
        # antisense: AAAC revcomp(spacer) [C if a G was added, to pair with it]
        top = fwd_oh + ("G" if add_g else "") + sp_in
        bot = rev_oh + rc(sp_in) + ("C" if add_g else "")
        if add_g:
            st.info("Your spacer does not start with G. A 5′ G was added for efficient U6 "
                    "transcription (21 nt total, which is normal and works).")
        st.code(f"sense      5'-{top}-3'\nantisense  5'-{bot}-3'", language=None)
        st.caption(f"Anneal the pair and ligate into {enzyme}-digested "
                   f"{vec.split(' (')[0]}. Standard desalted oligos are fine; use T4 PNK in the "
                   "ligation if they are not phosphorylated.")
        st.markdown(f"""
**Spacer:** {len(sp_in)} nt, GC {gc(sp_in):.0f}%{', **TTTT present** — may terminate U6 transcription' if 'TTTT' in sp_in else ''}

**Also order:** a U6 sequencing primer to verify the clone (`GAGGGCCTATTTCCCATGATT`).

**Before you order:** run this spacer through CRISPOR or CHOPCHOP for the genome-wide
off-target check, and confirm the oligo overhangs against your vector's own protocol.
""")
    else:
        st.info("Pick a spacer from the list, or paste one.")
