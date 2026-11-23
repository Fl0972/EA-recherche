"""
Calcul parallèle avec un timeout PAR TÂCHE.

joblib et multiprocessing.Pool ne savent pas interrompre une tâche trop
longue. Ici chaque processus de calcul reçoit une tâche à la fois ; le
processus principal note l'heure de départ et, si la tâche dépasse le délai,
tue le processus, en relance un neuf et renvoie un résultat « timeout ».
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time


def _travailleur(f, q_in, q_out):
    """Boucle d'un processus de calcul : (id, tâche) -> (id, résultat ou exception)."""
    while True:
        item = q_in.get()
        if item is None:
            return
        i, t = item
        try:
            q_out.put((i, "ok", f(t)))
        except Exception as e:                                   # noqa: BLE001
            q_out.put((i, "exception", repr(e)))


class _Proc:
    def __init__(self, ctx, f, q_out):
        self.q_in = ctx.Queue()
        self.p = ctx.Process(target=_travailleur, args=(f, self.q_in, q_out), daemon=True)
        self.p.start()
        self.tache = None
        self.debut = None


def map_timeout(f, taches, n_proc, timeout, sur_resultat, sur_timeout, progression=None):
    """
    Applique f à chaque tâche dans n_proc processus.
    sur_resultat(i, résultat) et sur_timeout(i, tâche) sont appelés dans le
    processus principal dès qu'une tâche se termine (pour écrire au fil de l'eau).
    ``f`` doit être une fonction de module (importable par les processus fils).
    """
    ctx = mp.get_context("spawn")
    q_out = ctx.Queue()
    procs = [_Proc(ctx, f, q_out) for _ in range(n_proc)]
    a_faire = list(enumerate(taches))[::-1]
    restantes = len(taches)
    faites = dernier = 0
    t0 = time.perf_counter()
    try:
        while restantes:
            for k, pr in enumerate(procs):                         # distribuer
                if pr.tache is None and a_faire:
                    i, t = a_faire.pop()
                    pr.tache, pr.debut = (i, t), time.perf_counter()
                    pr.q_in.put((i, t))
            try:                                                   # récupérer
                i, statut, res = q_out.get(timeout=0.5)
                for pr in procs:
                    if pr.tache is not None and pr.tache[0] == i:
                        tache = pr.tache[1]
                        pr.tache = None
                        break
                else:
                    continue
                if statut == "ok":
                    sur_resultat(i, res)
                else:
                    sur_resultat(i, {"__exception__": res, **tache})
                restantes -= 1
                faites += 1
            except queue.Empty:
                pass
            maintenant = time.perf_counter()
            for k, pr in enumerate(procs):                         # surveiller les délais
                if pr.tache is not None and maintenant - pr.debut > timeout:
                    i, t = pr.tache
                    pr.p.kill()
                    pr.p.join()
                    sur_timeout(i, t)
                    restantes -= 1
                    faites += 1
                    procs[k] = _Proc(ctx, f, q_out)
            if progression and faites >= dernier + progression:
                dernier = faites
                el = time.perf_counter() - t0
                print(f"    {faites}/{len(taches)} tâches, {el / 60:.1f} min écoulées, "
                      f"reste ~ {el / faites * restantes / 60:.1f} min", flush=True)
    finally:
        for pr in procs:
            try:
                pr.q_in.put(None)
            except Exception:                                      # noqa: BLE001
                pass
        for pr in procs:
            pr.p.join(timeout=5)
            if pr.p.is_alive():
                pr.p.kill()
