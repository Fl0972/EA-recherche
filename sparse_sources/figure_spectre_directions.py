"""Spectre des directions mesurées (jacobienne blanchie) : cercle seul vs cercle + grille 16x32.
Figure du compte rendu COMPTE_RENDU_METHODES.typ. Lancer : python sparse_sources/figure_spectre_directions.py"""
import os
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng=np.random.default_rng(0)
c=np.array([5.,10.]); R=np.sqrt(125)
def circle(M=64):
    t=2*np.pi*np.arange(M)/M; return c+R*np.c_[np.cos(t),np.sin(t)]
def grid(n,m):
    xs=(np.arange(n)+.5)*10/n; ys=(np.arange(m)+.5)*20/m
    X,Y=np.meshgrid(xs,ys); return np.c_[X.ravel(),Y.ravel()]
def draw(K):
    z=rng.uniform([0,0],[10,20],(K,2)); I=100*rng.dirichlet(np.ones(K)); return z,I
def jac(x,z,I):
    d=x[:,None,:]-z[None]; q=(d**2).sum(-1)
    phi=1/q; g=2*d/q[...,None]**2  # grad_z phi = 2(x-z)/q^2
    return np.hstack([phi, I*g[...,0], I*g[...,1]]), phi@I
def spec(x,z,I,sigma):
    J,y=jac(x,z,I); yb=y+5000/((x-c)**2).sum(1); s=sigma*yb
    return np.linalg.svd(J/s[:,None],compute_uv=False)
out={}
for K in [3,10,20,50]:
  for name,x in [("cercle 64",circle()),("cercle + grille 16x32",np.vstack([circle(),grid(16,32)]))]:
    cnt=[]
    for t in range(5):
      z,I=draw(K); sv=spec(x,z,I,1e-3); cnt.append(((1/sv)<0.1).sum())
      if K==50 and t==0: out[name]=sv
    print(f'K={K:2d}  {name:24s} directions a 0,1 pres : {np.median(cnt):.0f} / {3*K}')
plt.figure(figsize=(7,3.6))
for n,sv in out.items():
    v=np.full(150,np.nan); v[:len(sv)]=1/sv
    plt.semilogy(np.arange(1,151),v,'o-',ms=3,label=n)
plt.axvline(64.5,color='gray',ls='--',lw=1); plt.text(68,1e5,'au-delà de 64 : directions\nnon mesurées (cercle)',fontsize=8,color='gray')
plt.axhline(0.1,color='k',ls=':',label='précision 0,1')
plt.xlabel('direction j (triée)'); plt.ylabel(r'écart type sur la direction $1/s_j$')
plt.title(r'50 sources, $\sigma=10^{-3}$, source centrale : erreur minimale par direction')
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),'results','spectre_directions.png'),dpi=160)
# Poisson kernel check
x=circle(); z=np.array([[7.,15.]]); rho=np.linalg.norm(z-c); 
phi=1/((x-z)**2).sum(1); co=np.abs(np.fft.rfft(phi))/len(phi)
print("rho/R",rho/R, "ratio coeffs", co[2:7]/co[1:6])
