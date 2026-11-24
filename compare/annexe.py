"""
Annexe chiffrée des comptes rendus : toutes les valeurs tracées sur les figures,
recalculées depuis les CSV de results/ (mêmes médianes, quartiles et
proportions que compare/figures.py et compare/figures_deux_sources.py).

    python -m compare.annexe

Écrit :
    rapport/annexe_donnees.typ          inclus à la fin de rapport/compte_rendu_comparaison.typ
    rapport/compte_rendu_comparaison.md section « Annexe » remplacée entre deux marqueurs
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from compare import figures as F  # noqa: E402
from compare import figures_deux_sources as G  # noqa: E402
from compare import protocol as P  # noqa: E402
from compare import run as RUN  # noqa: E402

RDIR = ROOT / "results"
DDIR = RDIR / "deux_sources"
METHODES = ["prony", "ml_bic", "fw"]
NOM = {"prony": "Prony", "ml_bic": "ML + BIC", "fw": "Frank–Wolfe", "descente": "Descente",
       "gn": "Gauss–Newton"}
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
DEBUT, FIN = "<!-- ANNEXE : début (généré par python -m compare.annexe) -->", "<!-- ANNEXE : fin -->"


# ---- mise en forme -------------------------------------------------------------------------
def nb(v, chiffres=3):
    """Nombre à la française : 0,0932 ; 4,03·10⁵ ; 5,92·10⁻⁵."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    v = float(v)
    if v == 0:
        return "0"
    e = math.floor(math.log10(abs(v)))
    if -3 <= e <= 4:
        s = f"{v:.{max(chiffres - 1 - e, 0)}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s.replace(".", ",")
    m = f"{v / 10 ** e:.{chiffres - 1}f}".rstrip("0").rstrip(".").replace(".", ",")
    return f"{m}·10{str(e).translate(SUP)}"


def pc(p):
    return "—" if p is None or (isinstance(p, float) and math.isnan(p)) else f"{100 * p:.0f} %"


def mq(a):
    """médiane [Q1 ; Q3]"""
    a = np.asarray(a, float)
    if a.size == 0:
        return "—"
    return f"{nb(np.nanmedian(a))} [{nb(np.nanpercentile(a, 25))} ; {nb(np.nanpercentile(a, 75))}]"


class Doc:
    """Accumule le même contenu en Markdown et en Typst."""

    def __init__(self):
        self.md, self.ty = [], []

    def titre(self, niveau, t):
        self.md.append("#" * (niveau + 1) + " " + t + "\n")
        self.ty.append("=" * niveau + " " + esc(t) + "\n")

    def texte(self, t):
        self.md.append(t + "\n")
        self.ty.append(esc(t) + "\n")

    def table(self, entete, lignes, aligne=None):
        n = len(entete)
        aligne = aligne or ["l"] + ["r"] * (n - 1)
        self.md.append("| " + " | ".join(entete) + " |")
        self.md.append("|" + "|".join("---:" if a == "r" else "---" for a in aligne) + "|")
        for L in lignes:
            self.md.append("| " + " | ".join(str(c) for c in L) + " |")
        self.md.append("")
        al = ", ".join("right" if a == "r" else "left" for a in aligne)
        cells = ",\n  ".join(", ".join(f"[{esc(str(c))}]" for c in L) for L in lignes)
        head = ", ".join(f"[*{esc(h)}*]" for h in entete)
        self.ty.append(f"""#block(width: 100%, {{
  set par(justify: false)
  set text(size: 7.5pt, hyphenate: false)
  table(
  columns: {n}, align: ({al},), stroke: none, inset: (x: 3pt, y: 2.2pt),
  table.hline(),
  table.header({head}),
  table.hline(stroke: 0.5pt),
  {cells},
  table.hline(),
)}})
""")


def esc(t):
    for c in "\\*_#$[]<>@`~":
        t = t.replace(c, "\\" + c)
    return t


# ---- données -------------------------------------------------------------------------------
def lire_balayages():
    return {(b, c): F.lire(RDIR / f"{b}_{c}.csv") for b in RUN.BALAYAGES for c in P.CONFIGS}


def sel(rows, **kw):
    return [r for r in rows if all((np.isclose(r[k], v) if isinstance(v, float) else r[k] == v)
                                   for k, v in kw.items())]


def section_parametres(d):
    d.titre(2, "A.1 Paramètres communs à toutes les figures")
    g = RUN.grilles(False)
    lignes = [
        ["Hangar D", "[0, 10] × [0, 20]"],
        ["Centre c (source centrale)", "(5, 10)"],
        ["Cercle de capteurs", "64 capteurs, centre c, rayon √125 ≈ 11,18"],
        ["Grille de capteurs", "16 × 32 = 512 capteurs dans D (pas 0,625)"],
        ["Somme des intensités S", "100 (intensités ~ S · Dirichlet(1, …, 1))"],
        ["Nombres de sources N", "2, 4, 6, 10"],
        ["Bruit relatif σ (balayage a)", " / ".join(nb(s) for s in g["sigmas"])],
        ["I_c / S (balayage b)", " / ".join(nb(x) for x in g["ics"])],
        ["Cadre par défaut", "σ = 10⁻³, I_c = 50 S = 5 000"],
        ["Tirages par point", "100 (graine : (N, tirage, flux, 20261123))"],
        ["Erreur E", "(1/N) Σ (α d_i + β r_i), α = 0,75, β = 0,25"],
        ["Plafonds (source manquée)", "d = 5, r = 1"],
        ["Timeout", "60 s par tirage et par méthode ; les tirages arrêtés à 60 s ont été relancés une fois "
                    "avec 300 s (4 processus) ; 300 s dans le cas à deux sources"],
        ["Prony", "seuil de rang rank_tol = 0,005 ; capteurs du cercle seulement"],
        ["ML + BIC", "L-BFGS-B, ajout glouton, k_max = 20"],
        ["Frank–Wolfe glissant", "blanchi, noyau normalisé, λ = 1,5 √M, 40 itérations au plus"],
        ["Descente d'Orano", "20 000 itérations au plus, Armijo c = 10⁻⁴, pas initial 10⁻², "
                             "déplacement ≤ 5 % du domaine, intensités initiales S / N"],
        ["Cas à deux sources", "x = I₁ / I₂ ∈ {1, 3, 10, 30, 100, 300}, I₁ = S x / (x + 1), "
                               "I₂ = S / (x + 1) ; réussite : N̂ = 2 et d_max < 0,1"],
        ["Cadres à deux sources", "Orano : σ = 0, I_c = 0 ; projet : σ = 10⁻³, I_c = 50 S"],
        ["Réussite d'Orano (synthèse)", "95 / 85 / 55 / 20 / 5 % pour x = 1 / 3 / 10 / 30 / 100"],
    ]
    d.table(["Paramètre", "Valeur"], lignes, ["l", "l"])


def ic(r, base="E"):
    """moyenne [IC 95 %]"""
    return f"{nb(r[base + '_moyenne'])} [{nb(r[base + '_ic95_bas'])} ; {nb(r[base + '_ic95_haut'])}]"


def lire_agrege(b, c):
    p = RDIR / f"agrege_{b}_{c}.csv"
    if not p.exists():
        return []
    import csv
    with open(p, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in r:
            if k != "methode":
                r[k] = float(r[k]) if r[k] not in ("", "nan") else np.nan
    return rows


def section_balayage(d, b, xs, titre_fig, titre_x):
    for c in P.CONFIGS:
        rows = lire_agrege(b, c)
        cfg = "cercle" if c == "cercle" else "cercle + grille"
        d.titre(3, f"{titre_fig} — {cfg}")
        lignes = []
        for N in RUN.NS:
            for x in xs:
                for m in METHODES:
                    v = [r for r in rows if r["methode"] == m and r["N"] == N and np.isclose(r["x"], x)]
                    if not v:
                        continue
                    r = v[0]
                    lignes.append([str(N), nb(x), NOM[m], ic(r), nb(r["E_mediane"]), ic(r, "masse_parasite"),
                                   pc(r["N_exact"]), nb(r["E_ancienne_mediane"])])
        d.table(["N", titre_x, "Méthode", "E moyenne [IC 95 %]", "E médiane", "masse parasite [IC 95 %]",
                 "N̂ = N", "E ancienne (médiane)"], lignes, ["r", "r", "l"] + ["r"] * 5)


def section_nsources(d):
    d.titre(2, "A.4 Nombre de sources (fig_nsources)")
    d.texte("Cadre par défaut (σ = 10⁻³, I_c = 50 S). d et r : moyennes après fusion ; fausses : nombre moyen "
            "de sources estimées (après fusion) non appariées ; N̂ = N : nombre brut de sources estimées égal à N.")
    for c in P.CONFIGS:
        rows = lire_agrege("nsources", c)
        d.titre(3, "cercle" if c == "cercle" else "cercle + grille")
        lignes = []
        for m in METHODES:
            for N in RUN.NS:
                v = [r for r in rows if r["methode"] == m and r["N"] == N]
                if not v:
                    continue
                r = v[0]
                lignes.append([NOM[m], str(N), ic(r), nb(r["E_mediane"]), nb(r["d_moy_moyenne"]),
                               nb(r["r_moy_moyenne"]), ic(r, "masse_parasite"), nb(r["fausses_moyenne"], 2),
                               pc(r["N_exact"]), str(int(r["timeouts"]))])
        d.table(["Méthode", "N", "E moyenne [IC 95 %]", "E médiane", "d", "r", "masse parasite [IC 95 %]",
                 "fausses", "N̂ = N", "timeouts"], lignes, ["l"] + ["r"] * 9)


def section_delta(d):
    d.titre(2, "A.5 Sensibilité au rayon de fusion δ (fig_delta)")
    d.texte("Cadre par défaut (σ = 10⁻³, I_c = 50 S) ; δ = 0 : pas de fusion (ancienne métrique). "
            "La figure ne montre que cercle + grille, N = 4 et 10.")
    import csv
    for c in P.CONFIGS:
        p = RDIR / f"delta_{c}.csv"
        if not p.exists():
            continue
        with open(p, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for k in r:
                if k != "methode":
                    r[k] = float(r[k])
        d.titre(3, "cercle" if c == "cercle" else "cercle + grille")
        lignes = [[nb(r["delta"]), str(int(r["N"])), NOM[r["methode"]], ic(r), nb(r["E_mediane"]),
                   ic(r, "masse_parasite")]
                  for r in sorted(rows, key=lambda r: (r["N"], METHODES.index(r["methode"]), r["delta"]))]
        d.table(["δ", "N", "Méthode", "E moyenne [IC 95 %]", "E médiane", "masse parasite [IC 95 %]"],
                lignes, ["r", "r", "l", "r", "r", "r"])


def section_temps(d, R, brut):
    d.titre(2, "A.6 Temps de calcul (fig_nsources c)")
    d.texte("Temps par tirage en secondes, cadre par défaut : médiane [Q1 ; Q3] et maximum hors timeout. "
            "Mesuré avec 8 processus sur 4 cœurs (Intel Core i5-10210U).")
    lignes = []
    for c in P.CONFIGS:
        rows = R[("temps", c)]
        for m in METHODES:
            for N in RUN.NS:
                v = [r["temps"] for r in sel(rows, methode=m, N=N) if r["statut"] == "ok"]
                lignes.append(["cercle" if c == "cercle" else "cercle + grille", NOM[m], str(N), mq(v),
                               nb(max(v)) if v else "—"])
    d.table(["Disposition", "Méthode", "N", "temps (s)", "max (s)"], lignes, ["l", "l", "r", "r", "r"])
    d.titre(3, "Timeouts sur l'ensemble des balayages")
    lignes = []
    for c in P.CONFIGS:
        for m in METHODES:
            v = [r for r in brut[c] if r["methode"] == m]
            lignes.append(["cercle" if c == "cercle" else "cercle + grille", NOM[m], str(len(v)),
                           str(sum(r["statut"] == "timeout" for r in v)),
                           str(sum(r["statut"] == "exception" for r in v))])
    d.table(["Disposition", "Méthode", "calculs", "timeouts", "exceptions"], lignes, ["l", "l", "r", "r", "r"])


# ---- cas à deux sources --------------------------------------------------------------------
CADRES = [("orano", "cercle", "cas test d'Orano"), ("projet", "cercle", "projet, cercle"),
          ("projet", "grille", "projet, cercle + grille")]


def section_deux(d):
    R = G.lire(DDIR / "rapport.csv")
    XS = G.XS

    d.titre(2, "A.7 Réussite à deux sources (fig_deux_sources a et b)")
    d.texte(f"Réussite après fusion (δ = {nb(P.DELTA_DEF)}) : chaque vraie source appariée à moins de 0,1, "
            "quel que soit le nombre de sources estimées ; entre parenthèses, l'ancien critère (N̂ = 2 et chaque "
            "source à moins de 0,1). 100 tirages par point. « GN depuis la vérité » : Gauss–Newton parti des "
            "vraies sources (ce qu'une méthode parfaite atteindrait avec ce critère et ces données).")
    from compare.figures_synthese import lire_deux_sources, reussites
    F2 = reussites(lire_deux_sources(), P.DELTA_DEF)
    lignes = [["Orano (synthèse)", "descente"] + [pc(G.ORANO.get(x, np.nan)) for x in XS]]
    for c, cfg, txt in CADRES:
        rows = G.sel(R, cadre=c, config=cfg)
        rf = [r for r in F2 if r["cadre"] == c and r["config"] == cfg]
        for m, nom in list(zip(G.METH, [NOM[m] for m in G.METH])) + [("ref", "GN depuis la vérité")]:
            L = [txt, nom]
            for x in XS:
                vf = [r[m] for r in rf if float(r["x"]) == x]
                v = G.sel(rows, x=float(x))
                anc = G.taux(v, f"ok_{m}") if m != "ref" else np.mean([r["d_ref"] < 0.1 for r in v])
                L.append(f"{pc(np.mean(vf))} ({pc(anc)})")
            lignes.append(L)
    d.table(["Cadre", "Méthode"] + [f"x = {x}" for x in XS], lignes, ["l", "l"] + ["r"] * 6)

    d.titre(2, "A.8 Issue des descentes")
    d.texte("Nombre de descentes (sur 100) dans chaque catégorie.")
    lignes = []
    for c, cfg, txt in CADRES:
        rows = G.sel(R, cadre=c, config=cfg)
        for x in XS:
            v = G.sel(rows, x=float(x))
            lignes.append([txt, str(x)] + [str(sum(r["cat_descente"] == k for r in v)) for k in G.CAT])
    d.table(["Cadre", "x"] + G.CAT, lignes, ["l"] + ["r"] * (1 + len(G.CAT)))

    d.titre(2, "A.9 Conditionnement et itérations (fig_deux_sources c)")
    lignes = []
    for c, cfg, txt in CADRES:
        rows = G.sel(R, cadre=c, config=cfg)
        for x in XS:
            v = G.sel(rows, x=float(x))
            ok = [r["it_descente"] for r in v if r["ok_descente"]]
            lignes.append([txt, str(x), mq([r["kappa"] for r in v]), mq([r["it_descente"] for r in v]),
                           nb(np.median(ok)) if ok else "—", pc(np.mean([r["it_descente"] >= 20000 for r in v])),
                           mq([r["d_descente"] for r in v])])
    d.table(["Cadre", "x", "κ à la vérité", "itérations", "itér. (réussites)", "budget atteint",
             "d_max descente"], lignes, ["l"] + ["r"] * 6)
    ro = G.sel(R, cadre="orano", config="cercle")
    kap = [np.median([r["kappa"] for r in G.sel(ro, x=float(x))]) for x in XS]
    pente = np.polyfit(np.log(XS[2:]), np.log(kap[2:]), 1)[0]
    pente_tout = np.polyfit(np.log(XS), np.log(kap), 1)[0]
    ok = [r["kappa"] for r in ro if r["ok_descente"]]
    d.texte(f"Pente de log κ en fonction de log x (cas test d'Orano, médianes) : {nb(pente)} pour x ≥ 10, "
            f"{nb(pente_tout)} sur tous les x (Orano : κ ≥ c x², pente 2). Descentes réussies : 90 % ont "
            f"κ < {nb(np.percentile(ok, 90))}, κ maximal {nb(max(ok))}.")

    d.titre(2, "A.10 Paysage de la perte")
    situation_paysage(d, R)

    d.titre(2, "A.11 Bassins d'attraction (ancien critère)")
    B = G.lire(DDIR / "bassins.csv")
    with open(DDIR / "bassins_situations.json", encoding="utf-8") as f:
        sits = json.load(f)
    lignes = []
    for s in sits:
        v = G.sel(B, cadre=s["cadre"], x=float(s["x"]), tirage=float(s["tirage"]))
        sit = situation(s["cadre"], s["x"], s["tirage"], "cercle")
        r0 = G.sel(R, cadre=s["cadre"], config="cercle", x=float(s["x"]), tirage=float(s["tirage"]))[0]
        lignes.append([f"{s['cadre']}, x = {s['x']}, tirage {s['tirage']}", pos(sit.z), pos(depart(sit)[1:]),
                       nb(r0["separation"]), nb(r0["kappa"]), str(len(v))]
                      + [str(sum(r["cat_descente"] == k for r in v)) for k in ("réussite", "budget épuisé",
                                                                               "source éteinte")]
                      + [str(sum(bool(r["ok_gn"]) for r in v))])
    d.table(["Situation", "vraies sources", "départ fixe source 2", "distance", "κ", "départs",
             "réussite", "budget épuisé", "source éteinte", "GN réussit"], lignes, ["l", "l", "l"] + ["r"] * 7)
    d.texte("Les départs de la source 1 forment une grille régulière de 12 × 24 points au centre des cases "
            "de D (pas 10/12 en abscisse, 20/24 en ordonnée).")

    d.titre(2, "A.12 Distance entre les sources (ancien critère)")
    bins = [0, 2, 4, 8, 23]
    lignes = []
    for x in (1, 30):
        rows = G.sel(R, cadre="orano", config="cercle", x=float(x))
        for lo, hi in zip(bins[:-1], bins[1:]):
            v = [r for r in rows if lo <= r["separation"] < hi]
            lignes.append([str(x), f"[{lo}, {hi})", str(len(v))] + [pc(G.taux(v, f"ok_{m}")) for m in G.METH])
    d.table(["x", "distance", "tirages"] + [NOM[m] for m in G.METH], lignes, ["r", "l"] + ["r"] * 6)

    d.titre(2, "A.13 Temps à deux sources")
    d.texte("Temps médian par tirage, en secondes.")
    lignes = []
    for c, cfg, txt in CADRES:
        rows = G.sel(R, cadre=c, config=cfg)
        for m in G.METH:
            lignes.append([txt, NOM[m]] + [nb(np.median([r[f"t_{m}"] for r in G.sel(rows, x=float(x))]))
                                           for x in XS])
    d.table(["Cadre", "Méthode"] + [f"x = {x}" for x in XS], lignes, ["l", "l"] + ["r"] * 6)


def situation(cadre, x, tirage, config):
    from compare import deux_sources as DS
    sig, ic = DS.CADRES[cadre]
    return DS.situation_x(float(x), int(tirage), sig, ic, config)


def depart(sit):
    return P.LO + sit.u0 * (P.HI - P.LO)


def pos(z):
    return " et ".join(f"({nb(p[0], 3)} ; {nb(p[1], 3)})" for p in np.atleast_2d(z))


def situation_paysage(d, R):
    """Recalcule les deux descentes tracées (même situation, même départ que la figure)."""
    from compare.deux_sources import CADRES as CAD, choisir_situation
    from sparse_sources.orano import Probleme, descente_projetee
    t, _ = choisir_situation(R, "orano", 30.0)
    sig, ic = CAD["orano"]
    sit0 = situation("orano", 30, t, "cercle")
    d.texte(f"Cas test d'Orano (σ = 0, sans source centrale), x = 30, tirage {t}. Vraies sources : "
            f"{pos(sit0.z)}, intensités {', '.join(nb(i) for i in sit0.I)} ; distance "
            f"{nb(np.linalg.norm(sit0.z[0] - sit0.z[1]))}. Départ commun de la descente : "
            f"{pos(depart(sit0))}, intensités initiales 50 et 50. Carte : perte sur une grille "
            f"101 × 201 de positions de la source faible, l'autre source et les intensités à la vérité.")
    lignes = []
    for cfg in ("cercle", "grille"):
        sit = situation("orano", 30, t, cfg)
        pb = Probleme(sit.X, P.LO, P.HI, sit.m)
        r = descente_projetee(pb, sit.u0, np.full(2, 0.5), iter_max=20000)
        dist = np.linalg.norm(r["p"][:, None, :] - sit.z[None], axis=-1).min(axis=0)
        u, a = pb.normalise(sit.z, sit.I)
        lignes.append(["cercle" if cfg == "cercle" else "cercle + grille", str(r["iterations"]),
                       pos(r["p"]), " / ".join(nb(i) for i in r["I"]), nb(dist[0]), nb(dist[1]),
                       nb(r["perte"]), nb(pb.perte(u, a))])
    d.table(["Disposition", "itérations", "positions finales", "intensités finales",
             "dist. source forte", "dist. source faible", "perte finale", "perte à la vérité"],
            lignes, ["l", "r", "l", "l", "r", "r", "r", "r"])


# ---- assemblage ----------------------------------------------------------------------------
def construire():
    R = lire_balayages()
    brut = {c: RUN.lire_sources(RUN.chemin_sources(c)).values() for c in P.CONFIGS}
    brut = {c: [r for r in v if r["methode"] in METHODES] for c, v in brut.items()}
    g = RUN.grilles(False)
    d = Doc()
    d.titre(1, "Annexe : valeurs numériques des figures")
    d.texte("Toutes les valeurs ci-dessous sont recalculées depuis results/ (sources estimées de chaque tirage "
            "dans results/sources/) par python -m compare.annexe ; ce sont exactement celles qui sont tracées.")
    d.texte(f"Métrique (δ = {nb(P.DELTA_DEF)}) : les sources estimées, puis les vraies sources, sont fusionnées "
            "par regroupement hiérarchique à lien simple au seuil δ (intensité = somme, position = barycentre "
            "pondéré par les intensités) ; appariement hongrois ; E = (1/N') Σ (0,75 d_i + 0,25 r_i) sur les N' "
            "vraies sources fusionnées, d_i ≤ 5, r_i ≤ 1, source manquée = (5, 1) ; le nombre de sources estimé "
            "n'entre pas dans E. Masse parasite (à part) = somme des intensités des sources estimées non "
            "appariées / S. Agrégation : moyenne sur les 100 tirages et intervalle de confiance à 95 % par "
            "bootstrap (2 000 rééchantillonnages) ; la médiane est donnée à titre de comparaison, ainsi que "
            "la médiane de l'ancienne métrique (sans fusion, médiane). « N̂ = N » : proportion de tirages où le "
            "nombre brut de sources estimées vaut N. Médiane [Q1 ; Q3] : temps et conditionnement.")
    section_parametres(d)
    d.titre(2, "A.2 Balayage en bruit (fig_balayages, ligne du haut)")
    d.texte("I_c = 50 S.")
    section_balayage(d, "bruit", g["sigmas"], "Bruit", "σ")
    d.titre(2, "A.3 Balayage en source centrale (fig_balayages, ligne du bas)")
    d.texte("σ = 10⁻³.")
    section_balayage(d, "centrale", g["ics"], "Source centrale", "I_c / S")
    section_nsources(d)
    section_delta(d)
    section_temps(d, R, brut)
    if (DDIR / "rapport.csv").exists() and (RDIR / "sources" / "deux_sources.jsonl").exists():
        section_deux(d)
    return d


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    d = construire()
    typ = ("// Annexe générée par python -m compare.annexe — ne pas modifier à la main.\n"
           + "\n".join(d.ty))
    (ROOT / "rapport" / "annexe_donnees.typ").write_text(typ, encoding="utf-8")
    md_path = ROOT / "rapport" / "compte_rendu_comparaison.md"
    md = md_path.read_text(encoding="utf-8")
    bloc = DEBUT + "\n\n" + "\n".join(d.md) + "\n" + FIN
    if DEBUT in md:
        md = md[:md.index(DEBUT)] + bloc + md[md.index(FIN) + len(FIN):]
    else:
        md = md.rstrip() + "\n\n---\n\n" + bloc + "\n"
    md_path.write_text(md, encoding="utf-8")
    print("annexe écrite : rapport/annexe_donnees.typ et rapport/compte_rendu_comparaison.md")


if __name__ == "__main__":
    main()
