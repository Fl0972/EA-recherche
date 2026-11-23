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
    table.cell(rowspan: 4)[Prony], [2], [1,29], [1,57], [0,494], [10 %], [0,0014 s], [1,29], [1,57], [0,494], [10 %], [0,0015 s],
    [4], [1,34], [1,59], [0,504], [10 %], [0,0012 s], [1,34], [1,59], [0,504], [10 %], [0,0013 s],
    [6], [2,34], [2,9], [0,732], [20 %], [0,0013 s], [2,34], [2,9], [0,732], [20 %], [0,0014 s],
    [10], [2,6], [3,17], [0,875], [20 %], [0,0013 s], [2,6], [3,17], [0,875], [20 %], [0,0013 s],
    table.cell(rowspan: 4)[ML + BIC], [2], [0,185], [0,192], [0,164], [80 %], [0,18 s], [0,000404], [0,000318], [0,000805], [100 %], [0,54 s],
    [4], [1,23], [1,49], [0,44], [30 %], [0,43 s], [0,00104], [0,000729], [0,00192], [80 %], [0,98 s],
    [6], [1,69], [2,07], [0,611], [0 %], [0,48 s], [0,0031], [0,00269], [0,0047], [90 %], [1,8 s],
    [10], [2,53], [3,09], [0,829], [0 %], [0,83 s], [0,0132], [0,0103], [0,0172], [70 %], [5,6 s],
    table.cell(rowspan: 4)[Frank–Wolfe glissant], [2], [0,43], [0,527], [0,138], [70 %], [0,06 s], [0,000713], [0,000613], [0,00111], [90 %], [0,37 s],
    [4], [1,79], [2,15], [0,566], [30 %], [0,12 s], [0,00521], [0,00283], [0,0121], [80 %], [0,58 s],
    [6], [2,34], [2,89], [0,783], [0 %], [0,12 s], [0,0329], [0,0131], [0,0961], [20 %], [1,2 s],
    [10], [2,79], [3,43], [0,869], [0 %], [0,16 s], [0,0963], [0,0594], [0,113], [30 %], [3 s],
    table.hline(),
  )),
  kind: table,
  caption: [Cadre par défaut ($I_c = 50 S$, $sigma = 10^(-3)$), médianes sur les tirages :
    erreur $E$, erreur de position moyenne $d$ (unités du hangar), erreur relative d'intensité
    moyenne $r$, proportion de tirages où $hat(N) = N$, temps de calcul. La descente de
    gradient reçoit le vrai $N$ ; Prony n'utilise que les 64 capteurs du cercle.],
) <tab:synthese>
