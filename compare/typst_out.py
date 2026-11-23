"""
Sorties Typst de la comparaison :
    figures/comparaison/synthese.typ   tableau de synthèse (valeurs calculées depuis les CSV)
    rapport/comparaison.typ            squelette de la partie « Comparaison » : titres,
                                       figures, légendes factuelles, et sous chaque figure
                                       un commentaire « // À rédiger : » avec des
                                       observations chiffrées tirées des CSV.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from compare.figures import lire
from compare.protocol import METHODES as TOUTES, NOMS

METHODES = list(TOUTES)          # méthodes présentes (fixé par ecrire())

NS = [2, 4, 6, 10]
ROOT = Path(__file__).resolve().parents[1]


# ---- petites statistiques -------------------------------------------------------------
def sel(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if (np.isclose(r[k], v) if isinstance(v, float) else r[k] == v)]
    return out


def med(rows, champ="E"):
    v = [r[champ] for r in rows]
    return float(np.nanmedian(v)) if v else np.nan


def pex(rows):
    return float(np.mean([r["N_exact"] for r in rows])) if rows else np.nan


def fr(x, nd=3):
    """Nombre au format français, nd chiffres significatifs."""
    if x is None or not np.isfinite(x):
        return "—"
    if x == 0:
        return "0"
    if abs(x) >= 1e5:
        m, e = f"{x:.{nd}g}".split("e") if "e" in f"{x:.{nd}g}" else (f"{x:.{nd}e}".split("e"))
        return f"{float(m):.{nd}g}".replace(".", ",") + f" × 10^{int(e)}"
    if abs(x) >= 1000:
        s = f"{x:.0f}"
    else:
        s = f"{x:.{nd}g}"
        if "e" in s:
            m, e = s.split("e")
            return f"{m.replace('.', ',')} × 10^{int(e)}"
    return s.replace(".", ",")


def pc(x):
    return "—" if not np.isfinite(x) else f"{100 * x:.0f} %"


def meilleure(rows, N, champ="E", **kw):
    """(méthode, valeur) de plus petite médiane."""
    vals = {m: med(sel(rows, methode=m, N=N, **kw), champ) for m in METHODES}
    vals = {m: v for m, v in vals.items() if np.isfinite(v)}
    m = min(vals, key=vals.get)
    return m, vals[m]


# ---- tableau de synthèse -------------------------------------------------------------
def tableau(rdir: Path):
    R = {c: lire(rdir / f"nsources_{c}.csv") for c in ("cercle", "grille")}
    cfgs = [c for c in ("cercle", "grille") if R[c]]
    lignes = []
    entete_cfg = ", ".join(f"table.cell(colspan: 5)[*{ {'cercle': 'cercle (64)', 'grille': 'cercle + grille (576)'}[c] }*]"
                           for c in cfgs)
    sous = ", ".join("[$E$], [$d$], [$r$], [$hat(N) = N$], [temps]" for _ in cfgs)
    for m in METHODES:
        for k, N in enumerate(NS):
            cells = [f"table.cell(rowspan: {len(NS)})[{NOMS[m]}]"] if k == 0 else []
            cells.append(f"[{N}]")
            for c in cfgs:
                s = sel(R[c], methode=m, N=N)
                cells += [f"[{fr(med(s, 'E'))}]", f"[{fr(med(s, 'd_moy'))}]", f"[{fr(med(s, 'r_moy'))}]",
                          "[—]" if m == "descente" else f"[{pc(pex(s))}]",
                          f"[{fr(med(s, 'temps'), 2)} s]"]
            lignes.append(", ".join(cells))
    n_col = 2 + 5 * len(cfgs)
    desc = "La descente de gradient reçoit le vrai $N$ ; " if "descente" in METHODES else ""
    corps = ",\n    ".join(lignes)
    return f"""// Généré par python -m compare.run (compare/typst_out.py) à partir de results/nsources_*.csv.
// Ne pas modifier à la main : relancer « python -m compare.run --figures ».
#figure(
  text(size: 7.5pt, table(
    columns: {n_col},
    align: (left, center) + (right,) * {n_col - 2},
    stroke: none,
    table.hline(),
    table.header(
      table.cell(rowspan: 2)[*méthode*], table.cell(rowspan: 2)[*$N$*], {entete_cfg},
      {sous},
    ),
    table.hline(stroke: 0.5pt),
    {corps},
    table.hline(),
  )),
  kind: table,
  caption: [Cadre par défaut ($I_c = 50 S$, $sigma = 10^(-3)$), médianes sur les tirages :
    erreur $E$, erreur de position moyenne $d$ (unités du hangar), erreur relative d'intensité
    moyenne $r$, proportion de tirages où $hat(N) = N$, temps de calcul. {desc}Prony n'utilise que les 64 capteurs du cercle.],
) <tab:synthese>
"""


# ---- observations chiffrées -------------------------------------------------------------
def obs_bruit(rows, config):
    o = []
    sig = sorted({r["sigma"] for r in rows})
    for N in (2, 10):
        m, v = meilleure(rows, N, sigma=1e-3)
        o.append(f"N = {N}, σ = 10^-3 : meilleure E médiane = {fr(v)} ({NOMS[m]}) ; "
                 + ", ".join(f"{NOMS[k]} {fr(med(sel(rows, methode=k, N=N, sigma=1e-3)))}"
                             for k in METHODES if k != m))
    ml = [(s, pex(sel(rows, methode="ml_bic", N=4, sigma=s))) for s in sig]
    bas = [s for s, p in ml if p < 0.5]
    o.append("ML + BIC, N = 4 : proportion N̂ = N de "
             + " → ".join(f"{pc(p)}" for _, p in ml) + f" quand σ va de {fr(sig[0])} à {fr(sig[-1])}"
             + (f" (sous 50 % dès σ = {fr(bas[0])})" if bas else ""))
    o.append("N = 2, E médiane à σ = " + fr(sig[0]) + " et σ = " + fr(sig[-1]) + " : " + " ; ".join(
        f"{NOMS[k]} {fr(med(sel(rows, methode=k, N=2, sigma=sig[0])))} → "
        f"{fr(med(sel(rows, methode=k, N=2, sigma=sig[-1])))}" for k in METHODES))
    return o


def obs_centrale(rows, config):
    o = []
    ics = sorted({r["ic_ratio"] for r in rows})
    for N in (2, 6):
        o.append(f"N = {N}, ML + BIC : E médiane {fr(med(sel(rows, methode='ml_bic', N=N, ic_ratio=ics[0])))}"
                 f" sans source centrale (I_c/S = {ics[0]:g}) et "
                 f"{fr(med(sel(rows, methode='ml_bic', N=N, ic_ratio=ics[-1])))} pour I_c/S = {ics[-1]:g}")
    o.append("N = 4, proportion N̂ = N pour I_c/S = " + ", ".join(f"{x:g}" for x in ics) + " : "
             + " ; ".join(f"{NOMS[m]} " + "/".join(pc(pex(sel(rows, methode=m, N=4, ic_ratio=x)))
                                                   for x in ics)
                          for m in ("ml_bic", "fw", "prony")))
    return o


def obs_nsources(rows, config):
    o = []
    for m in METHODES:
        o.append(f"{NOMS[m]} : E médiane " + ", ".join(
            f"{fr(med(sel(rows, methode=m, N=N)))} (N = {N})" for N in NS)
            + ("" if m == "descente" else " ; N̂ = N : " + ", ".join(pc(pex(sel(rows, methode=m, N=N)))
                                                                  for N in NS)))
    pr = [np.mean([r["fausses"] if isinstance(r["fausses"], (int, float)) else float(r["fausses"])
                   for r in sel(rows, methode="prony", N=N)]) for N in NS]
    o.append("Prony : nombre moyen de fausses sources " + ", ".join(f"{fr(v, 2)} (N = {N})"
                                                                    for v, N in zip(pr, NS)))
    return o


def obs_temps(rows, config):
    o = []
    for m in METHODES:
        o.append(f"{NOMS[m]} : temps médian " + ", ".join(
            f"{fr(med(sel(rows, methode=m, N=N), 'temps'), 2)} s (N = {N})" for N in NS))
    to = {m: sum(r["statut"] == "timeout" for r in sel(rows, methode=m)) for m in METHODES}
    o.append("tirages en timeout (cadre par défaut) : " + ", ".join(f"{NOMS[m]} {v}" for m, v in to.items()))
    return o


def obs_synthese(rc, rg, champ_x, x_def):
    o = []
    for m in [k for k in ("ml_bic", "fw", "descente") if k in METHODES]:
        for N in (2, 10):
            a = med(sel(rc, methode=m, N=N, **{champ_x: x_def}))
            b = med(sel(rg, methode=m, N=N, **{champ_x: x_def}))
            o.append(f"{NOMS[m]}, N = {N}, {'σ' if champ_x == 'sigma' else 'I_c/S'} = {fr(x_def)} : "
                     f"E médiane {fr(a)} (cercle) → {fr(b)} (cercle + grille)")
    return o


# ---- squelette --------------------------------------------------------------------------
def bloc_figure(chemin, etiquette, legende, obs):
    lignes = "\n".join(f"//   - {x}" for x in obs)
    return f"""#figure(
  image("{chemin}", width: 100%),
  caption: [{legende}],
) <{etiquette}>
// À rédiger :
{lignes}
"""


def ecrire(rdir: Path, fdir: Path, args):
    METHODES[:] = [m for m in TOUTES if m in args.methodes]
    rel = os.path.relpath(fdir, ROOT / "rapport").replace("\\", "/")
    R = {(b, c): lire(rdir / f"{b}_{c}.csv") for b in ("bruit", "centrale", "nsources", "temps")
         for c in ("cercle", "grille")}
    if not any(R.values()):
        return
    (fdir / "synthese.typ").write_text(tableau(rdir), encoding="utf-8")
    n_tir = 1 + max(int(r["tirage"]) for r in R[("nsources", "cercle")]) if R[("nsources", "cercle")] else 0
    a, b = args.alpha, args.beta
    fr_ab = (fr(a).replace(",", '","'), fr(b).replace(",", '","'))   # virgule décimale en mode math

    def section_config(c, titre):
        lab = "cercle" if c == "cercle" else "grille"
        cap_cfg = ("64 capteurs $s_k$ sur le cercle de centre $c$ passant par les coins de $D$"
                   if c == "cercle" else
                   "64 capteurs sur le cercle et 512 capteurs en grille $16 times 32$ dans $D$ "
                   "(Prony n'utilise que les capteurs du cercle)")
        out = [f"== {titre}\n"]
        if R[("bruit", c)]:
            out.append(bloc_figure(
                f"{rel}/bruit_{c}.pdf", f"fig:bruit-{lab}",
                f"Erreur $E$ (médiane et quartiles, échelle log) et proportion de tirages où "
                f"$hat(N) = N$ en fonction du bruit relatif $sigma$, pour $I_c = 50 S$ ; {cap_cfg}. "
                f"{n_tir} tirages par point.",
                obs_bruit(R[("bruit", c)], c)))
        if R[("centrale", c)]:
            out.append(bloc_figure(
                f"{rel}/centrale_{c}.pdf", f"fig:centrale-{lab}",
                f"Erreur $E$ et proportion de $hat(N) = N$ en fonction de l'intensité de la source "
                f"centrale $I_c \\/ S$ (0 : pas de source centrale), $sigma = 10^(-3)$ ; {cap_cfg}.",
                obs_centrale(R[("centrale", c)], c)))
        if R[("nsources", c)]:
            out.append(bloc_figure(
                f"{rel}/nsources_{c}.pdf", f"fig:nsources-{lab}",
                f"Erreur $E$ et proportion de $hat(N) = N$ en fonction du nombre de sources $N$, "
                f"cadre par défaut ($I_c = 50 S$, $sigma = 10^(-3)$) ; {cap_cfg}.",
                obs_nsources(R[("nsources", c)], c)))
        if R[("temps", c)]:
            out.append(bloc_figure(
                f"{rel}/temps_{c}.pdf", f"fig:temps-{lab}",
                f"Temps de calcul par tirage (médiane et quartiles, génération des données exclue) "
                f"en fonction de $N$, cadre par défaut ; {cap_cfg}.",
                obs_temps(R[("temps", c)], c)))
        return "\n".join(out)

    synth = []
    if R[("bruit", "cercle")] and R[("bruit", "grille")]:
        synth.append("=== Cercle contre cercle et grille\n")
        synth.append(bloc_figure(
            f"{rel}/synthese_bruit.pdf", "fig:synthese-bruit",
            "Balayage en bruit : cercle seul (traits pleins) et cercle + grille (pointillés), "
            "mêmes tirages.", obs_synthese(R[("bruit", "cercle")], R[("bruit", "grille")], "sigma", 1e-3)))
        synth.append(bloc_figure(
            f"{rel}/synthese_centrale.pdf", "fig:synthese-centrale",
            "Balayage en source centrale : cercle seul (traits pleins) et cercle + grille (pointillés).",
            obs_synthese(R[("centrale", "cercle")], R[("centrale", "grille")], "ic_ratio", 0.0)))
        synth.append(bloc_figure(
            f"{rel}/synthese_nsources_temps.pdf", "fig:synthese-nsources",
            "Erreur $E$, proportion de $hat(N) = N$ et temps médian en fonction de $N$, cadre par "
            "défaut : cercle seul (traits pleins) et cercle + grille (pointillés).",
            obs_synthese_n(R[("nsources", "cercle")], R[("nsources", "grille")])))

    ds = "" if args.quick else section_deux_sources()
    renvoi = " (@fig:ds-reussite)" if "<fig:ds-reussite>" in ds else ""
    ligne_descente = ("- La descente de gradient reçoit le vrai $N$ ; Prony, ML + BIC et Frank–Wolfe estiment $N$."
                      if "descente" in METHODES else
                      "- Prony, ML + BIC et Frank–Wolfe estiment $N$. La descente de gradient d'Orano (20 000 "
                      "itérations, $N$ connu) est trop lente pour ces balayages ; elle est étudiée à part dans le "
                      f"cas à 2 sources{renvoi}.")
    typ = f"""// Partie « Comparaison » du rapport — squelette généré par python -m compare.run
// (compare/typst_out.py). Les observations « À rédiger » sont calculées depuis results/*.csv.
// Les figures sont en PDF (Typst ≥ 0.14 sait inclure des PDF ; sinon remplacer .pdf par .png).

== Le cas test d'Orano

#figure(
  rect(width: 100%, height: 6cm, fill: luma(230), stroke: none),
  caption: [Cas test d'Orano à deux sources (figure à fournir).],
) <fig:orano>
// Résultats existants (synthèse Orano) :
//   - Orano a retenu l'approche la plus simple, une descente de gradient sur θ = (z_i, I_i),
//     testée uniquement sur 2 sources.
//   - Réussite 95 %, 85 %, 55 %, 20 %, 5 % pour un rapport d'intensité x = 1, 3, 10, 30, 100.
//   - Ralentissement dû au conditionnement : κ ≥ c x².
//   - Gauss–Newton : 90 % à x = 100, en 59 ms.
//   - Prony + ML : 100 % jusqu'à x = 300, en 10 à 25 ms.
{ds}
== Protocole de comparaison

- Situations : $N in {{2, 4, 6, 10}}$ sources $z_i$ uniformes dans $D$, intensités $I_i$ aléatoires
  (Dirichlet) de somme $S = 100$ ; mêmes situations, mêmes bruits et mêmes graines pour toutes les
  méthodes et les deux dispositions de capteurs.
- Mesures : $y_k^"brut" = Phi_theta (s_k) + I_c \\/ norm(s_k - c)^2$, bruit relatif
  $y_k^"brut" (1 + sigma epsilon_k)$, source centrale retirée ; $sigma_k = sigma y_k^"brut"$.
- Cadre par défaut : $I_c = 50 S$, $sigma = 10^(-3)$.
- Appariement des sources estimées et vraies par l'algorithme hongrois sur la distance euclidienne.
- Erreur d'un tirage : $E = 1/N sum_i (alpha d_i + beta r_i)$, $alpha = {fr_ab[0]}$, $beta = {fr_ab[1]}$,
  $d_i = norm(hat(z)_i - z_i)$, $r_i = abs(hat(I)_i - I_i) \\/ I_i$ ; source manquée : $d_i = 5$,
  $r_i = 1$ (plafonds) ; les fausses sources sont comptées à part.
{ligne_descente}
- {n_tir} tirages par point ; timeout de {fr(args.timeout)} s par tirage et par méthode (compté comme échec).

{section_config("cercle", "Capteurs sur le cercle")}
{section_config("grille", "Cercle et grille de capteurs")}
{"".join(synth)}
== Synthèse

#include "{rel}/synthese.typ"
"""
    nom = "comparaison_quick.typ" if args.quick else "comparaison.typ"
    (ROOT / "rapport" / nom).write_text(typ, encoding="utf-8")
    print(f"  Typst : rapport/{nom}, {fdir.relative_to(ROOT)}/synthese.typ")


def obs_synthese_n(rc, rg):
    o = []
    for m in METHODES:
        o.append(f"{NOMS[m]} : E médiane à N = 10 {fr(med(sel(rc, methode=m, N=10)))} (cercle) → "
                 f"{fr(med(sel(rg, methode=m, N=10)))} (grille) ; temps médian "
                 f"{fr(med(sel(rc, methode=m, N=10), 'temps'), 2)} s → {fr(med(sel(rg, methode=m, N=10), 'temps'), 2)} s")
    return o


# ---- section « cas à deux sources » (compare.deux_sources) ------------------------------
def section_deux_sources():
    """Figures et observations de results/deux_sources/ ; chaîne vide si absent."""
    import json
    from compare import figures_deux_sources as F
    rd = ROOT / "results" / "deux_sources"
    if not (rd / "rapport.csv").exists():
        return ""
    R = F.lire(rd / "rapport.csv")
    XS = F.XS
    rel = "../figures/deux_sources"
    nom = {"descente": "descente", "gn": "Gauss–Newton", "ml_bic": "ML + BIC", "fw": "Frank–Wolfe",
           "prony": "Prony"}
    n_tir = len(F.sel(R, cadre="orano", config="cercle", x=1.0))

    def t(c, cfg, m, x):
        return F.taux(F.sel(R, cadre=c, config=cfg, x=float(x)), f"ok_{m}")

    o1 = []
    for c, cfg, txt in (("orano", "cercle", "cas test d'Orano"), ("projet", "cercle", "projet, cercle"),
                        ("projet", "grille", "projet, cercle + grille")):
        o1.append(f"{txt} : réussite pour x = " + ", ".join(map(str, XS)) + " — " + " ; ".join(
            f"{nom[m]} " + "/".join(pc(t(c, cfg, m, x)) for x in XS) for m in F.METH))
    o1.append("synthèse Orano (descente) : 95/85/55/20/5 % pour x = 1, 3, 10, 30, 100")

    o2 = []
    for c, cfg, txt in (("orano", "cercle", "cas test d'Orano"), ("projet", "cercle", "projet, cercle"),
                        ("projet", "grille", "projet, cercle + grille")):
        rows = F.sel(R, cadre=c, config=cfg)
        o2.append(f"{txt}, tous x confondus : " + ", ".join(
            f"{k} {pc(np.mean([r['cat_descente'] == k for r in rows]))}" for k in F.CAT
            if any(r["cat_descente"] == k for r in rows)))
    ro = F.sel(R, cadre="orano", config="cercle")
    o2.append("cas test d'Orano, itérations de la descente (médiane) : " + ", ".join(
        f"{np.median([r['it_descente'] for r in F.sel(ro, x=float(x))]):.0f} (x = {x})" for x in XS))

    kap = [np.median([r["kappa"] for r in F.sel(ro, x=float(x))]) for x in XS]
    pente = np.polyfit(np.log(XS[2:]), np.log(kap[2:]), 1)[0]
    ok = [r for r in ro if r["ok_descente"]]
    o3 = [f"κ médian à la vérité : " + ", ".join(f"{fr(k)} (x = {x})" for k, x in zip(kap, XS)),
          f"pente log κ / log x pour x ≥ 10 : {fr(pente)} (Orano : κ ≥ c x², pente 2)",
          f"descente réussie : itérations médianes {np.median([r['it_descente'] for r in ok]):.0f} ; "
          f"budget de 20 000 atteint dans {pc(np.mean([r['it_descente'] >= 20000 for r in ro]))} des tirages"
          if ok else "aucune descente réussie"]

    o4 = ["carte : perte d'Orano quand la source faible est déplacée (le reste à la vérité) ; sur le cercle "
          "la perte varie lentement (vallée large et plate), la descente s'approche vite puis rampe "
          "(budget de 20 000 itérations atteint) ; sur cercle + grille la perte est hérissée d'un pic "
          "sur chaque capteur de la grille (singularité en 1/r²) avec un plateau entre les pics : "
          "la descente s'arrête au bout de quelques centaines d'itérations, loin des vraies sources"]

    o5 = []
    if (rd / "bassins.csv").exists():
        B = F.lire(rd / "bassins.csv")
        with open(rd / "bassins_situations.json", encoding="utf-8") as f:
            for s in json.load(f):
                rows = F.sel(B, cadre=s["cadre"], x=float(s["x"]), tirage=float(s["tirage"]))
                o5.append(f"{s['cadre']}, x = {s['x']}, tirage {s['tirage']} : la descente réussit depuis "
                          f"{pc(np.mean([r['cat_descente'] == 'réussite' for r in rows]))} des départs, "
                          f"Gauss–Newton depuis {pc(np.mean([r['ok_gn'] for r in rows]))}")

    bins = [0, 2, 4, 8, 23]
    o6 = []
    for x in (1, 30):
        rows = F.sel(ro, x=float(x))
        o6.append(f"x = {x}, descente : réussite " + ", ".join(
            f"{pc(F.taux([r for r in rows if lo <= r['separation'] < hi], 'ok_descente'))} "
            f"(distance [{lo}, {hi}))" for lo, hi in zip(bins[:-1], bins[1:])))

    rp = F.sel(R, cadre="projet", config="cercle")
    o7 = ["projet, cercle, temps médian à x = 1 : " + ", ".join(
        f"{nom[m]} {fr(np.median([r[f't_{m}'] for r in F.sel(rp, x=1.0)]), 2)} s" for m in F.METH)]

    blocs = [
        ("ds_reussite_rapport", "fig:ds-reussite",
         f"Réussite à 2 sources ($hat(N) = 2$ et chaque source à moins de 0,1) en fonction du rapport "
         f"d'intensité $x = I_1 \\/ I_2$, {n_tir} tirages par point. Gauss–Newton et la descente reçoivent "
         f"$N = 2$ et partent du même point. Croix : valeurs de la synthèse Orano.", o1),
        ("ds_causes", "fig:ds-causes",
         "Issue de chaque descente : réussite, source éteinte (intensité projetée à 0), budget de "
         "20 000 itérations épuisé, limite statistique (perte finale au plus égale à celle du minimum "
         "voisin de la vérité), source bloquée au bord de $D$, minimum local.", o2),
        ("ds_conditionnement", "fig:ds-conditionnement",
         "À gauche : conditionnement $kappa$ de $J^top J$ (perte d'Orano, paramètres normalisés) aux "
         "vraies sources, en fonction de $x$. À droite : itérations de la descente en fonction de "
         "$kappa$, cas test d'Orano.", o3),
        ("ds_paysage", "fig:ds-paysage",
         "Perte d'Orano en fonction de la position de la source faible (l'autre source et les "
         "intensités à leur vraie valeur) et trajectoire de la descente, sur le cercle et sur "
         "cercle + grille ; à droite, distance de chaque source estimée à la vraie source la plus "
         "proche au fil des itérations.", o4),
        ("ds_bassins", "fig:ds-bassins",
         "Bassins d'attraction : issue de la descente selon la position de départ de la source 1 "
         "(la source 2 part toujours du même point) ; croix : départs depuis lesquels Gauss–Newton "
         "échoue.", o5),
        ("ds_separation", "fig:ds-separation",
         "Réussite en fonction de la distance entre les deux sources, cas test d'Orano, $x = 1$ et "
         "$x = 30$.", o6),
        ("ds_temps", "fig:ds-temps",
         "Temps de calcul médian par tirage en fonction de $x$.", o7),
    ]
    if not (ROOT / "figures" / "deux_sources" / "ds_reussite_rapport.pdf").exists():
        return ""
    out = ["\n== Le cas à deux sources\n"]
    for f, lab, cap, obs in blocs:
        if (ROOT / "figures" / "deux_sources" / f"{f}.pdf").exists():
            out.append(bloc_figure(f"{rel}/{f}.pdf", lab, cap, obs))
    return "\n".join(out)
