"""
Construit la présentation courte d'avancement (PDF 16:9, 5 transparents) :
fait tourner les simulations (figures.py), puis assemble les transparents.
Les figures plus détaillées servent aussi à la note d'avancement.

    python presentation/build_presentation.py            # tout recalculer
    python presentation/build_presentation.py --reuse    # réutiliser les figures existantes
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

import figures as F

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "avancement_2026-11-12.pdf"
W, H = 13.333, 7.5
NAVY, TEXT, MUTED, ORANGE = "#1F3A5F", "#222222", "#6B7280", "#E07A1F"
TITLE_BAND = 0.115


# ---- Mise en page -------------------------------------------------------------
def _visible_len(token: str) -> float:
    if "$" not in token:
        return len(token)
    plain = re.sub(r"\\[a-zA-Z]+", "x", token)
    return len(re.sub(r"[${}_^\\]", "", plain)) * 0.9


def wrap(text: str, max_chars: float) -> list[str]:
    """Coupe en lignes sans jamais couper une formule $...$."""
    tokens = re.findall(r"(?:\$[^$]*\$|[^\s$])+", text)
    lines, line, length = [], [], 0.0
    for tok in tokens:
        n = _visible_len(tok)
        if line and length + 1 + n > max_chars:
            lines.append(" ".join(line))
            line, length = [], 0.0
        line.append(tok)
        length += n + (1 if length else 0)
    if line:
        lines.append(" ".join(line))
    return lines


class Deck:
    def __init__(self, path: Path, preview_dir: Path | None = None):
        self.pdf = PdfPages(path)
        self.page = 0
        self.preview_dir = preview_dir

    def slide(self, title: str, kicker: str | None = None):
        self.page += 1
        fig = plt.figure(figsize=(W, H))
        fig.patch.set_facecolor("white")
        fig.add_artist(Rectangle((0, 1 - TITLE_BAND), 1, TITLE_BAND,
                                 transform=fig.transFigure, color=NAVY))
        if kicker:
            fig.text(0.035, 1 - 0.032, kicker.upper(), color="#AFC4DD", fontsize=10,
                     va="center", weight="bold")
        fig.text(0.035, 1 - TITLE_BAND / 2 - (0.012 if kicker else 0), title,
                 color="white", fontsize=23, weight="bold", va="center")
        fig.text(0.035, 0.022, "Localisation de sources — point d'avancement",
                 color=MUTED, fontsize=9)
        fig.text(0.965, 0.022, str(self.page), color=MUTED, fontsize=9, ha="right")
        return fig

    def save(self, fig):
        self.pdf.savefig(fig)
        if self.preview_dir:
            self.preview_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.preview_dir / f"slide_{self.page:02d}.png", dpi=55)
        plt.close(fig)

    def close(self):
        d = self.pdf.infodict()
        d["Title"] = "Localisation de sources de particules — point d'avancement"
        d["Author"] = "Florian Guillaumey"
        self.pdf.close()


def bullets(fig, items, x, y, width, size=15, spacing=1.45, gap=0.45):
    """
    items : "texte" (puce), ("-", "texte") (sous-puce), ("h", "titre") (titre
    en gras), ("n", "1.", "texte") (numéroté). Renvoie le y final.
    """
    line_h = size * spacing / 72 / H
    for item in items:
        kind, text, mark = "b", item, "•"
        if isinstance(item, tuple):
            kind = item[0]
            text = item[-1]
            mark = {"-": "–", "h": "", "n": item[1] if len(item) == 3 else ""}.get(kind, "•")
        indent = 0.022 if kind == "-" else 0.0
        fs = size - 2 if kind == "-" else size
        weight = "bold" if kind == "h" else "normal"
        color = NAVY if kind == "h" else TEXT
        text_x = x + indent + ((0.03 if kind == "n" else 0.02) if mark else 0)
        max_chars = (width - (text_x - x)) * W * 72 / (fs * 0.56)
        if kind == "h":
            y -= line_h * 0.25
        for k, line in enumerate(wrap(text, max_chars)):
            if k == 0 and mark:
                fig.text(x + indent, y, mark, fontsize=fs, color=ORANGE if kind != "-" else MUTED,
                         va="top", weight="bold")
            fig.text(text_x, y, line, fontsize=fs, color=color, va="top", weight=weight)
            y -= line_h * (fs / size)
        y -= line_h * gap * (fs / size)
    return y


def image(fig, path, x, y, w, h):
    ax = fig.add_axes([x, y, w, h])
    ax.imshow(plt.imread(path))
    ax.axis("off")
    return ax


def boxed(fig, x, y, w, h, text, size=14, color=NAVY):
    """Encadré de message clé."""
    fig.add_artist(Rectangle((x, y), w, h, transform=fig.transFigure,
                             facecolor="#EEF3F9", edgecolor=color, lw=1.2))
    lines = wrap(text, w * W * 72 / (size * 0.56) - 2)
    line_h = size * 1.4 / 72 / H
    y0 = y + h / 2 + line_h * (len(lines) - 1) / 2
    for k, line in enumerate(lines):
        fig.text(x + 0.015, y0 - k * line_h, line, fontsize=size, color=color,
                 va="center", weight="bold")


def formula(fig, tex, x, y, size=18, ha="left"):
    fig.text(x, y, tex, fontsize=size, color=TEXT, ha=ha, va="center")


# ---- Transparents -------------------------------------------------------------
# Version courte (5 transparents) pour une présentation partagée avec d'autres
# pistes ; le détail est dans la note d'avancement (document partagé).
def build(figs: dict, bench: dict, preview_dir=None):
    deck = Deck(OUTPUT, preview_dir)
    top = 1 - TITLE_BAND - 0.06
    mle = {r["sweep_value"]: r["n_accuracy"] for r in bench["3.0"]
           if r["method"].startswith("Prony +")}

    # 1. Titre ---------------------------------------------------------------
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(NAVY)
    deck.page += 1
    fig.text(0.07, 0.62, "Piste : reconstruction analytique", color="white",
             fontsize=34, weight="bold")
    fig.text(0.07, 0.535, "Localiser des sources de particules avec des capteurs sur un cercle",
             color="white", fontsize=22)
    fig.add_artist(Rectangle((0.07, 0.47), 0.12, 0.006, transform=fig.transFigure,
                             color=ORANGE))
    fig.text(0.07, 0.17, "Florian Guillaumey", color="white", fontsize=16, weight="bold")
    fig.text(0.07, 0.12, "12 novembre 2026", color="#C9D6E6", fontsize=14)
    deck.save(fig)

    # 2. Problème --------------------------------------------------------------
    fig = deck.slide("Le problème")
    fig.text(0.04, top, "Retrouver le nombre, la position et l'intensité des sources "
             "à partir de 64 mesures sur un cercle", fontsize=17, color=TEXT, va="top")
    formula(fig, r"$\Phi(x) = \sum_i \frac{I_i}{\|x - x_i\|^2}$", 0.5, top - 0.11,
            size=24, ha="center")
    image(fig, figs["situation"], 0.03, 0.06, 0.94, 0.58)
    deck.save(fig)

    # 3. Idée clé ---------------------------------------------------------------
    fig = deck.slide("Sur un cercle, une formule exacte… mais fragile")
    image(fig, figs["prony_bruit"], 0.02, 0.08, 0.6, 0.76)
    bullets(fig, [
        "Les coefficients de Fourier des mesures forment une somme d'exponentielles : "
        "la méthode de Prony retrouve les sources exactement, sans bruit",
        "Problème très mal posé : dès 0,1 % de bruit, une source faible proche du "
        "centre disparaît",
    ], 0.64, top, 0.33, size=17, gap=0.9)
    deck.save(fig)

    # 4. Résultat ---------------------------------------------------------------
    fig = deck.slide("Localiser : optimal. Compter : difficile")
    image(fig, figs["n_correct"], 0.02, 0.08, 0.6, 0.76)
    bullets(fig, [
        "Quand le nombre de sources est connu, Prony + maximum de vraisemblance atteint "
        "la précision optimale (borne de Cramér-Rao)",
        f"Le point faible est le comptage : {mle[0.01]:.0%} de réussite avec 3 sources "
        "en moyenne et 1 % de bruit",
    ], 0.64, top, 0.33, size=17, gap=0.9)
    deck.save(fig)

    # 5. Suite ---------------------------------------------------------------------
    fig = deck.slide("Suite et questions")
    bullets(fig, [
        "Entraîner un modèle d'IA pour estimer le nombre de sources : jeux de données "
        "et banc d'essai sont prêts",
        "Valider le modèle de bruit et le rôle de la source centrale",
        "Tester d'autres placements de capteurs",
    ], 0.06, top - 0.04, 0.86, size=20, gap=1.0)
    fig.text(0.06, 0.1, "Détails, hypothèses et résultats complets : note d'avancement "
             "(document partagé)", fontsize=13, color=MUTED, style="italic")
    deck.save(fig)

    deck.close()
    return OUTPUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true",
                        help="réutiliser les figures et résultats déjà calculés")
    parser.add_argument("--preview", help="dossier où enregistrer un aperçu PNG par page")
    args = parser.parse_args()

    cache = F.FIG_DIR / "results.json"
    if args.reuse and cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        print("Simulations et figures...")
        F.situation()
        F.prony_noise()
        F.fourier_decay()
        _, known, ratio = F.central_source()
        _, missed = F.missed_sources()
        _, bench = F.benchmark()
        data = {"central": [None, known, ratio], "missed": missed,
                "bench": {str(k): v for k, v in bench.items()}}
        cache.write_text(json.dumps(data, indent=1), encoding="utf-8")

    F.n_correct(data["bench"])
    figs = {name: F.FIG_DIR / f for name, f in [
        ("situation", "situation.png"), ("prony_bruit", "prony_bruit.png"),
        ("n_correct", "n_correct.png")]}
    path = build(figs, data["bench"], Path(args.preview) if args.preview else None)
    print(f"Présentation : {path}")


if __name__ == "__main__":
    main()
