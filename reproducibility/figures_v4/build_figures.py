from pathlib import Path
import csv,json,hashlib,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,Rectangle
P=Path(__file__).resolve().parents[2]; W=Path(__file__).resolve().parent
R=P.parents[1];O=P/'figures/v4';O.mkdir(parents=True,exist_ok=True)
D=P/'data/numerics';D.mkdir(parents=True,exist_ok=True)
F=R/'results/criticality_followup_20261026'
names=['late_summaries.csv','parent_windows.csv','purification_curves.csv','purification_parents.csv','purification_summary.csv','finite_grid_S.csv','lifetime_growth.csv','convergence_checks.csv','final_decision.json','config.json','independent_check.json']
pro=[]
for name in names:
 src=F/name; dst=D/name
 if src.exists():shutil.copy2(src,dst)
 else:src=dst
 pro.append({'source':str(src),'copy':str(dst.relative_to(P)),'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
(D/'SOURCE_HASHES.json').write_text(json.dumps(pro,indent=2))
def read(name):
 with (D/name).open() as f:return list(csv.DictReader(f))
blue='#0072B2';orange='#D55E00';green='#009E73';grey='#6B7280'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.4,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
def panel(ax,label):ax.text(-.16,1.04,label,transform=ax.transAxes,weight='bold',size=10)
def save(fig,name):
 fig.savefig(O/(name+'.pdf'),bbox_inches='tight');fig.savefig(O/(name+'.png'),bbox_inches='tight');plt.close(fig)
# Geometric schematic; no experimental points.
fig,ax=plt.subplots(1,2,figsize=(7.0,2.6),gridspec_kw={'width_ratios':[1.1,1]});
a=ax[0];a.set_xlim(-.5,7);a.set_ylim(-1.8,2.2);a.axis('off');panel(a,'(a)')
for j in range(7):
 a.plot(j,0,'o',color=blue,ms=5);a.text(j,-.37,str(j),ha='center')
a.plot(range(7),[0]*7,color=grey,lw=.8,zorder=0)
for j,y,col in [(1,1,orange),(2,1.6,green)]:
 a.plot([j,j+2],[y,y],lw=3,color=col);a.text(j+1,y+.15,r'$m_{%d}$'%j,ha='center');
 for k in range(j,j+3):a.plot([k,k],[0,y],ls=':',lw=.7,color=col)
a.axvline(3.5,ls='--',color='k',lw=.8,ymin=.2,ymax=.9);a.text(4.1,.8,r'$A\mid\bar A$');a.text(3,-1.15,r'$h_j\ \to\ T(c)^{\otimes3}\ \to\ m_j\ \to\ P_{j,\pm}$',ha='center');a.text(3,-1.65,'Three letters updated before measurement',ha='center',fontsize=8)
a=ax[1];a.set_xlim(-.5,6.7);a.set_ylim(-1.6,2);a.axis('off');panel(a,'(b)')
coords={0:(0,.7),1:(1,.7),2:(2,.7),3:(3.5,.7),4:(4.5,1.25),5:(5.5,.7)}
for j,(x,y) in coords.items():a.plot(x,y,'o',ms=8,color=green if j>=3 else blue);a.text(x,y-.4,str(j),ha='center')
for i,j in [(0,1),(1,2),(3,4),(4,5),(3,5)]:
 x,y=coords[i];u,v=coords[j];a.plot([x,u],[y,v],color=grey,zorder=0)
a.axvline(2.75,ls='--',color=orange,lw=1);a.text(1,.1,'finite component',ha='center');a.text(4.5,-.05,'isolated triangle',ha='center');a.text(2.8,1.6,'no anticommuting edge',ha='center',color=orange);a.text(3,-1.2,'Vertices: anchors     Edges: anticommutation',ha='center',fontsize=8)
fig.tight_layout(w_pad=3);save(fig,'model_geometry')
# Exact probabilities and exact-count checks.
q=np.linspace(0,.5,501);pi=np.array([q,q,1-2*q]);u2=(pi**2).sum(0);u3=(pi**3).sum(0);rho=u3**2+(u2-u3)**2;piso=6*pi.prod(0)*u3**4+2*u3**2*(u2-u3)**3
np.savetxt(O/'exact_probabilities.csv',np.c_[q,rho,piso],delimiter=',',header='q,rho,p_iso',comments='')
fig,ax=plt.subplots(1,2,figsize=(7,2.8));ax[0].plot(q,rho,color=blue);ax[0].axhline(1/18,ls=':',color=grey,label=r'$1/18$');ax[0].legend(frameon=False,loc='upper right');ax[0].set(xlabel=r'$q$',ylabel=r'Cut probability $\rho$');ax[1].plot(q,piso,color=green);ax[1].set(xlabel=r'$q$',ylabel=r'Isolated-triangle probability $p_{\rm iso}$');ax[1].ticklabel_format(axis='y',style='sci',scilimits=(-2,-2))
for a in ax:
 for val in [.13,1/3]:a.axvline(val,color=grey,lw=.6,ls='--')
for a,l in zip(ax,['(a)','(b)']):panel(a,l)
inax=ax[1].inset_axes([.53,.51,.43,.43]);mask=(q>=.1)&(q<=.16);inax.plot(q[mask],piso[mask],color=green);inax.plot(.13,.003813512380467423,'o',color='black',ms=3);inax.set(xticks=[.10,.13,.16],yticks=[.003,.004]);inax.tick_params(labelsize=6);inax.set_title(r'$0.10\leq q\leq0.16$',fontsize=7,pad=2)
for val in [0,.13,1/3,.5]:
 pp=np.array([val,val,1-2*val]);uu2=(pp**2).sum();uu3=(pp**3).sum();rr=uu3**2+(uu2-uu3)**2;ii=6*pp.prod()*uu3**4+2*uu3**2*(uu2-uu3)**3
 ax[0].plot(val,rr,'o',ms=4,mfc='white',mec=blue,clip_on=False);ax[1].plot(val,ii,'o',ms=4,mfc='white',mec=green,clip_on=False)
fig.tight_layout(w_pad=3);save(fig,'exact_probabilities')
# Bounds: actual analytic bound, large sizes no trajectories.
pp=np.array([.13,.13,.74]);rr=((pp**3).sum())**2+((pp**2).sum()-(pp**3).sum())**2
L=np.unique(np.rint(np.geomspace(32,4096,200)/2).astype(int)*2);m=L//16-1;bound=8/rr-4+2*L*(1-rr)**m
fig,ax=plt.subplots(1,2,figsize=(7,2.8));ax[0].plot(L,np.minimum(L/2,bound),color=blue,label='finite-size bound');ax[0].plot(L,L/2,ls=':',color=grey,label=r'$L/2$');ax[0].axhline(8/rr-4,color=green,ls='--',label=r'$8/\rho-4$');ax[0].set(xscale='log',xlabel=r'$L$',ylabel=r'Upper bound on $\mathbb{E}\sup S_A$ (bits)',ylim=(0,120));ax[0].legend(frameon=False,fontsize=7)
tau=np.linspace(0,.45,200);ax[1].plot(tau,8*np.exp(4*tau)/rr-4,color=orange);ax[1].set(xlabel=r'$\tau=(1-c^3)t/L$',ylabel='Sufficient area-bound constant (bits)')
inax=ax[1].inset_axes([.15,.57,.4,.35]);inax.plot(tau,rr*np.exp(-4*tau),color=blue);inax.set(xlabel=r'$\tau$',ylabel=r'$p_t$',xticks=[0,.4],yticks=[.05,.15]);inax.tick_params(labelsize=6);inax.xaxis.label.set_size(7);inax.yaxis.label.set_size(7)
for a,l in zip(ax,['(a)','(b)']):panel(a,l)
fig.tight_layout(w_pad=3);save(fig,'refresh_bounds')
# Frozen numerical summaries, independent parent trajectories.
sums=read('late_summaries.csv');parents=read('parent_windows.csv');fig,ax=plt.subplots(1,2,figsize=(7,3));jit=np.random.default_rng(20261027)
for a,obs in zip(ax,['S','I3']):
 for c,col,mk,off in [(0,blue,'o',-.022),(.6,orange,'s',.022)]:
  rows=sorted([r for r in sums if float(r['q'])==.13 and float(r['c'])==c and r['observable']==obs],key=lambda r:int(r['L']))
  for r in rows:
   ll=int(r['L']);x=np.log2(ll)+off
   pp=[float(p['late_'+obs]) for p in parents if float(p['q'])==.13 and float(p['c'])==c and int(p['L'])==ll and p['initial']=='zero']
   assert len(pp)==128
   a.scatter(x+jit.uniform(-.045,.045,len(pp)),pp,s=3,color=col,alpha=.18,edgecolors='none',zorder=1)
  x=np.log2([int(r['L']) for r in rows])+off;y=np.array([float(r['mean']) for r in rows]);err=np.array([[float(r['mean'])-float(r['lower']) for r in rows],[float(r['upper'])-float(r['mean']) for r in rows]])
  a.errorbar(x,y,yerr=err,marker=mk,color=col,capsize=3,label=f'$c={c:g}$',ms=5,zorder=3)
 a.set(xticks=np.log2([24,48,96]),xticklabels=['24','48','96'],xlabel=r'$L$')
ax[0].set_ylabel(r'Half-chain entropy $S_A$ (bits)');ax[1].set_ylabel(r'$I_3$ (bits)');ax[1].axhline(0,color=grey,lw=.7,ls=':');ax[0].legend(frameon=False);ax[0].set_title(r'$q=0.13$, 8192 sweeps',fontsize=8,pad=8)
for a,l in zip(ax,['(a)','(b)']):panel(a,l)
fig.tight_layout(w_pad=3);save(fig,'finite_memory_sizes')
# Purification and empirical last-bit survival.
curves=read('purification_curves.csv');lifes=read('purification_parents.csv');fig,ax=plt.subplots(1,2,figsize=(7,3.15));styles={24:':',48:'--',96:'-'}
for c,col in [(0,blue),(.6,orange)]:
 for ll in [24,48,96]:
  rows=sorted([r for r in curves if float(r['q'])==.13 and float(r['c'])==c and int(r['L'])==ll and float(r['time_sweeps'])>0],key=lambda r:float(r['time_sweeps']))
  x=np.array([float(r['time_sweeps']) for r in rows]);y=np.array([float(r['mean'])/ll for r in rows]);lo=np.array([float(r['lower'])/ll for r in rows]);hi=np.array([float(r['upper'])/ll for r in rows]);ax[0].plot(x,y,color=col,ls=styles[ll],label=f'$c={c:g}, L={ll}$');ax[0].fill_between(x,lo,hi,color=col,alpha=.08)
ax[0].set(xscale='log',xlim=(.1,2048),xlabel='Time (sweeps)',ylabel=r'$\mathbb{E} S_{\rm mix}/L$');ax[0].legend(frameon=False,fontsize=6.5,ncol=2,loc='upper right')
meds={}
for c,col in [(0,blue),(.6,orange)]:
 rows=[r for r in lifes if float(r['q'])==.13 and float(r['c'])==c and int(r['L'])==96];assert len(rows)==128;assert all(r['complete']=='True' for r in rows)
 ts=np.sort([float(r['tau']) for r in rows]);x=np.r_[0,ts];y=np.r_[1,1-np.arange(1,len(ts)+1)/len(ts)];ax[1].step(x,y,where='post',color=col,label=f'$c={c:g}$')
 meds[c]=[np.quantile([float(r['tau']) for r in lifes if float(r['q'])==.13 and float(r['c'])==c and int(r['L'])==ll], .5, method='lower') for ll in [24,48,96]]
ax[1].set(xlabel=r'Last-bit lifetime $\tau$ (sweeps)',ylabel=r'Empirical $P(\tau>x)$',xlim=(0,1.03*max(float(r['tau']) for r in lifes if float(r['q'])==.13 and int(r['L'])==96)),ylim=(-.03,1.03));ax[1].legend(frameon=False,loc='upper right');ax[1].text(.52,.64,r'$L=96$'+'\n'+'128 parents / protocol',transform=ax[1].transAxes,fontsize=7)
inax=ax[1].inset_axes([.53,.22,.42,.32]);
for c,col in [(0,blue),(.6,orange)]:inax.plot([24,48,96],meds[c],'o-',color=col,ms=3)
inax.set(xlabel=r'$L$',ylabel=r'median $\tau$',xticks=[24,96],yticks=[10,20],ylim=(9,21));inax.tick_params(labelsize=6);inax.xaxis.label.set_size(7);inax.yaxis.label.set_size(7)
for a,l in zip(ax,['(a)','(b)']):panel(a,l)
fig.tight_layout(w_pad=3);save(fig,'purification_lifetimes')
# QPU all blocks and preselected contrasts.
h=json.loads((P/'reproducibility/echo27/ECHO27_RESULTS.json').read_text());keys=['BELL0','IDLE_T','ECHO_T','P5_T','SEP0'];labels=['Bell ref.','Idle','Echo','Five proj.','Separable'];fig,ax=plt.subplots(1,2,figsize=(7,3.2),gridspec_kw={'width_ratios':[1,1.1]})
for i,k in enumerate(keys):
 d=h['groups'][k];blocks=[(1+sum(b['axis_means'][k+'_W'+str(j)] for j in range(3)))/4 for b in h['per_block']];ax[0].scatter(i+np.linspace(-.12,.12,4),blocks,marker='.',s=20,color=grey,alpha=.7,zorder=2);ax[0].errorbar(i,d['mean'],yerr=d['radius'],fmt='o',color=blue if k!='ECHO_T' else orange,capsize=3,ms=5,zorder=3)
ax[0].set(xticks=range(5),xticklabels=labels,ylabel='Nominal score',ylim=(0,1.08));ax[0].tick_params(axis='x',rotation=25);ax[0].text(.04,.94,'4 blocks; 1024 shots / coordinate',transform=ax[0].transAxes,fontsize=7)
cs=[('primary_D','Primary $D$',1),('ECHO_minus_IDLE','Echo $-$ idle',1),('P5_minus_ECHO','Echo $-$ five projections',-1)]
for i,(k,lab,sign) in enumerate(cs):
 d=h['contrasts'][k];ax[1].errorbar(sign*d['mean'],2-i,xerr=d['radius'],fmt='s' if i==0 else 'o',color=grey if i==0 else orange,capsize=3,ms=5)
ax[1].axvline(0,color=grey,ls=':',lw=1);ax[1].set(yticks=[2,1,0],yticklabels=[x[1] for x in cs],xlabel='Score contrast',xlim=(-.13,.71),ylim=(-.5,2.5));ax[1].text(.03,.95,'Frozen simultaneous intervals',transform=ax[1].transAxes,fontsize=7);ax[1].text(.04,.70,'Unresolved',transform=ax[1].transAxes,fontsize=7,color=grey)
for a,l in zip(ax,['(a)','(b)']):panel(a,l)
fig.tight_layout(w_pad=3);save(fig,'hardware_controls')
# Copy one identical figure set into PRA; portable source/data are packaged.
(P/'pra/manuscript/figures').mkdir(parents=True,exist_ok=True)
for f in O.glob('*.pdf'):shutil.copy2(f,P/'pra/manuscript/figures'/f.name)
summary={'figures':6,'new_production_runs':0,'scatter_parents':768,'q13_records':768,'last_bit_censoring_q13':sum(r['complete']!='True' for r in lifes if float(r['q'])==.13),'medians_tau':meds,'CI_policy':'unchanged frozen trajectory intervals; empirical survival descriptive; QPU frozen simultaneous intervals','data_sources':pro}
(W/'FIGURE_BUILD.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='data_sources'},indent=2))
