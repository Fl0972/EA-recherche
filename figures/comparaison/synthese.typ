// Généré par python -m compare.run (compare/typst_out.py) à partir de results/nsources_*.csv.
// Ne pas modifier à la main : relancer « python -m compare.run --figures ».
#figure(
  text(size: 7.5pt, table(
    columns: 12,
    align: (left, center) + (right,) * 10,
    stroke: none,
    table.hline(),
    table.header(
      table.cell(rowspan: 2)[*méthode*], table.cell(rowspan: 2)[*$N$*], table.cell(colspan: 5)[*cercle (64)*], table.cell(colspan: 5)[*cercle + grille (576)*],
      [$E$], [$d$], [$r$], [$hat(N) = N$], [temps], [$E$], [$d$], [$r$], [$hat(N) = N$], [temps],
    ),
    table.hline(stroke: 0.5pt),
    table.cell(rowspan: 4)[Prony], [2], [0,453], [0,566], [0,149], [16 %], [0,0013 s], [0,453], [0,566], [0,149], [16 %], [0,0014 s],
    [4], [1,84], [2,25], [0,551], [19 %], [0,0013 s], [1,84], [2,25], [0,551], [19 %], [0,0013 s],
    [6], [2,36], [2,89], [0,746], [20 %], [0,0013 s], [2,36], [2,89], [0,746], [20 %], [0,0014 s],
    [10], [2,71], [3,35], [0,838], [2 %], [0,0013 s], [2,71], [3,35], [0,838], [2 %], [0,0014 s],
    table.cell(rowspan: 4)[ML + BIC], [2], [0,0932], [0,113], [0,0312], [92 %], [0,21 s], [0,00059], [0,000496], [0,00078], [99 %], [0,55 s],
    [4], [1,24], [1,51], [0,427], [23 %], [0,36 s], [0,00153], [0,00126], [0,00207], [93 %], [1,1 s],
    [6], [1,85], [2,26], [0,618], [0 %], [0,46 s], [0,00328], [0,00252], [0,00477], [93 %], [1,8 s],
    [10], [2,59], [3,18], [0,793], [0 %], [0,61 s], [0,00671], [0,00485], [0,0101], [81 %], [6,9 s],
    table.cell(rowspan: 4)[Frank–Wolfe glissant], [2], [0,302], [0,382], [0,0468], [82 %], [0,067 s], [0,000997], [0,000766], [0,00129], [89 %], [0,35 s],
    [4], [1,75], [2,2], [0,555], [17 %], [0,1 s], [0,00796], [0,00487], [0,0144], [57 %], [0,61 s],
    [6], [2,22], [2,71], [0,743], [1 %], [0,14 s], [0,0263], [0,014], [0,0547], [27 %], [1,1 s],
    [10], [2,76], [3,4], [0,828], [0 %], [0,16 s], [0,068], [0,0385], [0,123], [21 %], [3,6 s],
    table.hline(),
  )),
  kind: table,
  caption: [Cadre par défaut ($I_c = 50 S$, $sigma = 10^(-3)$), médianes sur les tirages :
    erreur $E$, erreur de position moyenne $d$ (unités du hangar), erreur relative d'intensité
    moyenne $r$, proportion de tirages où $hat(N) = N$, temps de calcul. Prony n'utilise que les 64 capteurs du cercle.],
) <tab:synthese>
