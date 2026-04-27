import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size':11, 'font.family':'sans-serif', 'axes.titlesize':13,
                     'axes.labelsize':12, 'figure.dpi':150, 'savefig.dpi':300, 'savefig.bbox':'tight'})
FIGDIR = "results/lookahead/final/figures"
os.makedirs(FIGDIR, exist_ok=True)

# FIG 1: Baseline staircase
models = [("GPT2S",80.3,84.6),("GPT2M",78.7,87.6),("GPT2XL",80.7,87.4),
          ("Py410",80.9,89.2),("Py1B",82.3,90.1),("Py1.4B",83.2,90.0),
          ("Py2.8B",83.2,91.1),("Santa",91.1,93.1),("CdLlama",88.6,94.5),
          ("Llama1B",86.6,93.7),("Llama1BI",83.6,94.3)]
x=np.arange(len(models)); w=0.35
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,7),height_ratios=[3,1],gridspec_kw={'hspace':0.3})
ax1.bar(x-w/2,[m[1] for m in models],w,label='Probe',color='#4C72B0',alpha=.85)
ax1.bar(x+w/2,[m[2] for m in models],w,label='Name+Params',color='#DD8452',alpha=.85)
ax1.set_ylabel('Accuracy (%)'); ax1.set_title('Baseline Staircase: Name+Params Beats Probe Everywhere')
ax1.set_xticks(x); ax1.set_xticklabels([m[0] for m in models],rotation=45,ha='right')
ax1.legend(); ax1.set_ylim(70,100); ax1.grid(axis='y',alpha=.3)
gaps=[m[1]-m[2] for m in models]
ax2.bar(x,gaps,0.6,color=['#C44E52' if g<-5 else '#CCB974' for g in gaps],alpha=.85)
ax2.set_ylabel('Gap (%)'); ax2.set_title('Gap (Probe - N+P): All Negative')
ax2.set_xticks(x); ax2.set_xticklabels([m[0] for m in models],rotation=45,ha='right')
ax2.axhline(0,color='k',lw=.8); ax2.set_ylim(-15,2); ax2.grid(axis='y',alpha=.3)
plt.savefig(f"{FIGDIR}/fig1_baseline_staircase.png"); plt.savefig(f"{FIGDIR}/fig1_baseline_staircase.pdf"); plt.close()
print("Fig 1 ✓")

# FIG 2: K decay (4 models)
fig,ax=plt.subplots(figsize=(8,5))
for name,gaps,c,m in [('Pythia-2.8B',[18.7,10.3,6.0,-1.2],'#4C72B0','o'),
                       ('Qwen-1.5B',[51.3,24.5,11.1,2.7],'#DD8452','s'),
                       ('GPT-J-6B',[30.9,15.8,5.2,4.9],'#55A868','^'),
                       ('Qwen-7B',[60.3,22.5,10.6,5.0],'#C44E52','D')]:
    ax.plot([1,2,3,5],gaps,'-'+m,color=c,label=name,lw=2,ms=8)
ax.axhline(0,color='gray',ls='--',alpha=.5); ax.fill_between([.5,5.5],-5,5,alpha=.1,color='gray')
ax.set_xlabel('Prediction Distance (K)'); ax.set_ylabel('Gap: Probe - Context (%)')
ax.set_title('Future Lens K Decay: Universal Across 4 Models'); ax.set_xticks([1,2,3,5])
ax.legend(); ax.grid(alpha=.3); ax.set_xlim(.5,5.5)
plt.savefig(f"{FIGDIR}/fig2_future_lens_k_decay.png"); plt.savefig(f"{FIGDIR}/fig2_future_lens_k_decay.pdf"); plt.close()
print("Fig 2 ✓")

# FIG 3: Spearman
beh=[0,0,0,26.7,32,38.7,32.7,41.3,42.7,34,34]
probe=[80.3,78.7,80.7,80.9,82.3,83.2,83.2,91.1,88.6,86.6,83.6]
gap=[-3.9,-8.9,-5.5,-8.5,-7.8,-6.8,-7.3,-2,-5.9,-7.1,-10.7]
names=['GPT2S','GPT2M','GPT2XL','Py410','Py1B','Py1.4B','Py2.8B','Santa','CdLl','Llama','LlI']
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5))
a1.scatter(beh,probe,c='#4C72B0',s=80,zorder=5)
for i,n in enumerate(names): a1.annotate(n,(beh[i],probe[i]),fontsize=7,xytext=(5,5),textcoords='offset points')
z=np.polyfit(beh,probe,1); a1.plot(np.linspace(-5,50),np.poly1d(z)(np.linspace(-5,50)),'--',color='#4C72B0',alpha=.5)
a1.set_xlabel('Behavioral (%)'); a1.set_ylabel('Probe (%)'); a1.set_title('rho=+0.940'); a1.grid(alpha=.3)
a2.scatter(beh,gap,c='#C44E52',s=80,zorder=5)
for i,n in enumerate(names): a2.annotate(n,(beh[i],gap[i]),fontsize=7,xytext=(5,5),textcoords='offset points')
a2.axhline(0,color='k',lw=.5); a2.set_xlabel('Behavioral (%)'); a2.set_ylabel('Gap (%)')
a2.set_title('rho=-0.078'); a2.grid(alpha=.3)
plt.suptitle('Capability Predicts Probe Signal, Not Planning',fontsize=14,y=1.02); plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig3_spearman.png"); plt.savefig(f"{FIGDIR}/fig3_spearman.pdf"); plt.close()
print("Fig 3 ✓")

# FIG 4: Training dynamics (3 scales)
fig, ax = plt.subplots(figsize=(10, 6))
for fname, label, color, marker in [
    ('dynamics_p1_pythia_2_8b_deduped.json', 'Pythia-2.8B', '#C44E52', 'D'),
    ('dynamics_p1_pythia_1b_deduped.json', 'Pythia-1B', '#4C72B0', 'o'),
    ('dynamics_p1_pythia_410m_deduped.json', 'Pythia-410M', '#55A868', 's'),
]:
    path = f"results/lookahead/final/{fname}"
    if not os.path.exists(path): continue
    data = json.load(open(path))
    cps = sorted(data['checkpoints'], key=lambda x: int(x['step'].replace('step','')))
    steps = [int(cp['step'].replace('step','')) for cp in cps]
    gaps = [cp['gap']*100 for cp in cps]
    ax.plot(steps, gaps, '-'+marker, color=color, label=label, lw=2, ms=7, alpha=0.85)
ax.axhline(0, color='gray', ls='--', alpha=.5)
ax.set_xlabel('Training Step'); ax.set_ylabel('Gap: Probe - Baseline (%)')
ax.set_title('Training Dynamics: Probe Gap Across 3 Pythia Scales')
ax.set_xscale('symlog', linthresh=100); ax.legend(); ax.grid(alpha=.3)
plt.savefig(f"{FIGDIR}/fig4_training_dynamics.png"); plt.savefig(f"{FIGDIR}/fig4_training_dynamics.pdf"); plt.close()
print("Fig 4 ✓")

# FIG 5: Probe vs behavioral
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for idx, (p1n, p2n, label) in enumerate([
    ('dynamics_p1_pythia_2_8b_deduped.json', 'dynamics_p2_pythia_2_8b_deduped.json', 'Pythia-2.8B'),
    ('dynamics_p1_pythia_1b_deduped.json', 'dynamics_p2_pythia_1b_deduped.json', 'Pythia-1B'),
    ('dynamics_p1_pythia_410m_deduped.json', 'dynamics_p2_pythia_410m_deduped.json', 'Pythia-410M'),
]):
    ax = axes[idx]
    p1 = json.load(open(f"results/lookahead/final/{p1n}"))
    p2 = json.load(open(f"results/lookahead/final/{p2n}"))
    beh_steps = {cp['step']: cp['behavioral']*100 for cp in p2['checkpoints']}
    probe_steps = {cp['step']: cp['best_probe']*100 for cp in p1['checkpoints']}
    steps_ordered = ['step0', 'step512', 'step4000', 'step32000', 'step143000']
    p_vals = [probe_steps.get(s, 0) for s in steps_ordered]
    b_vals = [beh_steps.get(s, 0) for s in steps_ordered]
    x = np.arange(len(steps_ordered)); w = 0.35
    ax.bar(x-w/2, p_vals, w, label='Probe', color='#4C72B0', alpha=0.85)
    ax.bar(x+w/2, b_vals, w, label='Behavioral', color='#DD8452', alpha=0.85)
    ax.set_title(label); ax.set_xticks(x)
    ax.set_xticklabels(['0', '512', '4K', '32K', '143K'], fontsize=9)
    ax.set_xlabel('Training Step')
    if idx == 0: ax.set_ylabel('Accuracy (%)'); ax.legend()
    ax.grid(axis='y', alpha=.3); ax.set_ylim(0, 100)
plt.suptitle('Probe Leads Behavioral: Models "Know" Before They Can "Do"', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig5_probe_vs_behavioral.png"); plt.savefig(f"{FIGDIR}/fig5_probe_vs_behavioral.pdf"); plt.close()
print("Fig 5 ✓")

# FIG 6: K decay during training
fig, ax = plt.subplots(figsize=(8, 5))
fl = json.load(open("results/lookahead/final/dynamics_p3_future_lens.json"))
colors = {'step0':'#AAAAAA','step512':'#CCB974','step4000':'#55A868','step32000':'#4C72B0','step143000':'#C44E52'}
markers = {'step0':'x','step512':'s','step4000':'^','step32000':'o','step143000':'D'}
for cp in fl['checkpoints']:
    step = cp['step']; k_gaps = []; k_vals = []
    for k in [1, 3, 5]:
        kd = cp.get(f'k{k}', {})
        if 'gap' in kd: k_vals.append(k); k_gaps.append(kd['gap']*100)
    if k_vals:
        ax.plot(k_vals, k_gaps, '-'+markers.get(step,'o'), color=colors.get(step,'gray'),
                label=step, lw=2, ms=8, alpha=0.85)
ax.axhline(0, color='gray', ls='--', alpha=.5)
ax.set_xlabel('Prediction Distance (K)'); ax.set_ylabel('Gap (%)')
ax.set_title('K Decay Develops During Training (Pythia-2.8B)')
ax.set_xticks([1, 3, 5]); ax.legend(title='Training step'); ax.grid(alpha=.3)
plt.savefig(f"{FIGDIR}/fig6_k_decay_training.png"); plt.savefig(f"{FIGDIR}/fig6_k_decay_training.pdf"); plt.close()
print("Fig 6 ✓")

# FIG 7: Domain spectrum
gptj = json.load(open("results/lookahead/final/overnight_phase1a_domains.json"))["gptj_domains_50"]
qwen = json.load(open("results/lookahead/final/overnight_complete.json"))["qwen7b_domains_50"]
domains = ['chain_of_thought','chain_of_thought_scrambled','chain_of_thought_nonmath',
           'free_prose','structured_prose','code','poetry']
labels_short = ['CoT\nmath','CoT\nscram','CoT\nnon-math','Free\nprose','Struct\nprose','Code','Poetry']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
g_k3 = [gptj.get(d,{}).get('k3',{}).get('gap',0)*100 for d in domains]
g_lo = [gptj.get(d,{}).get('k3',{}).get('gap_ci_lo',0)*100 for d in domains]
g_hi = [gptj.get(d,{}).get('k3',{}).get('gap_ci_hi',0)*100 for d in domains]
g_err = [[g-l for g,l in zip(g_k3,g_lo)], [h-g for g,h in zip(g_k3,g_hi)]]
x = np.arange(len(domains))
ax1.bar(x, g_k3, color='#4C72B0', alpha=0.85, yerr=g_err, capsize=3)
ax1.set_xticks(x); ax1.set_xticklabels(labels_short, fontsize=9)
ax1.set_ylabel('Gap at K=3 (%)'); ax1.set_title('GPT-J-6B'); ax1.grid(axis='y',alpha=.3); ax1.axhline(0,color='k',lw=.5)
q_k3 = [qwen.get(d,{}).get('k3',{}).get('gap',0)*100 for d in domains]
q_lo = [qwen.get(d,{}).get('k3',{}).get('gap_ci_lo',0)*100 for d in domains]
q_hi = [qwen.get(d,{}).get('k3',{}).get('gap_ci_hi',0)*100 for d in domains]
q_err = [[g-l for g,l in zip(q_k3,q_lo)], [h-g for g,h in zip(q_k3,q_hi)]]
ax2.bar(x, q_k3, color='#DD8452', alpha=0.85, yerr=q_err, capsize=3)
ax2.set_xticks(x); ax2.set_xticklabels(labels_short, fontsize=9)
ax2.set_ylabel('Gap at K=3 (%)'); ax2.set_title('Qwen-7B'); ax2.grid(axis='y',alpha=.3); ax2.axhline(0,color='k',lw=.5)
plt.suptitle('Domain Spectrum at K=3 (50 prompts, Bootstrap CIs)', fontsize=14, y=1.02); plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig7_domain_spectrum.png"); plt.savefig(f"{FIGDIR}/fig7_domain_spectrum.pdf"); plt.close()
print("Fig 7 ✓")

# FIG 8: Attn vs MLP
am = json.load(open("results/lookahead/final/overnight_phase1b_attnmlp.json"))["gptj_attn_mlp"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for idx, (dom, title) in enumerate([('chain_of_thought','CoT Math'),('code','Code'),('poetry','Poetry')]):
    ax = axes[idx]
    if dom not in am: continue
    d = am[dom]; lyrs=[]; resid=[]; attn=[]; mlp=[]
    for l, accs in sorted(d.items(), key=lambda x: int(x[0]) if x[0].isdigit() else -1):
        if not isinstance(accs, dict): continue
        lyrs.append(int(l)); resid.append(accs['resid']*100); attn.append(accs['attn']*100); mlp.append(accs['mlp']*100)
    ax.plot(lyrs,resid,'-o',color='#4C72B0',label='Residual',lw=2,ms=6)
    ax.plot(lyrs,attn,'-s',color='#DD8452',label='Attention',lw=2,ms=6)
    ax.plot(lyrs,mlp,'-^',color='#55A868',label='MLP',lw=2,ms=6)
    ax.set_xlabel('Layer'); ax.set_title(title)
    if idx==0: ax.set_ylabel('Accuracy (%)'); ax.legend(fontsize=8)
    ax.grid(alpha=.3)
plt.suptitle('Attention vs MLP (GPT-J-6B, K=3)', fontsize=14, y=1.02); plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig8_attn_mlp.png"); plt.savefig(f"{FIGDIR}/fig8_attn_mlp.pdf"); plt.close()
print("Fig 8 ✓")

# FIG 9: Transfer matrix
tr = json.load(open("results/lookahead/final/overnight_phase1c_transfer.json"))["gptj_transfer"]
if "transfer_matrix" in tr:
    doms = ["chain_of_thought","code","free_prose","poetry"]; labs = ["CoT","Code","Prose","Poetry"]
    matrix = np.zeros((4,4))
    for i, d1 in enumerate(doms):
        if d1 in tr["transfer_matrix"]:
            for j, d2 in enumerate(doms):
                matrix[i,j] = tr["transfer_matrix"][d1].get(d2, 0)*100
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=40, vmax=85, aspect='auto')
    ax.set_xticks(range(4)); ax.set_xticklabels(labs); ax.set_yticks(range(4)); ax.set_yticklabels(labs)
    ax.set_xlabel('Test Domain'); ax.set_ylabel('Train Domain')
    ax.set_title('Cross-Domain Transfer (GPT-J)\nDiag=within, Off-diag~chance(46%)')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{matrix[i,j]:.1f}', ha='center', va='center', fontsize=11,
                    fontweight='bold' if i==j else 'normal', color='white' if matrix[i,j]>65 else 'black')
    plt.colorbar(im, label='Accuracy (%)'); plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig9_transfer_matrix.png"); plt.savefig(f"{FIGDIR}/fig9_transfer_matrix.pdf"); plt.close()
    print("Fig 9 ✓")

print(f"\nTotal PNGs: {len([f for f in os.listdir(FIGDIR) if f.endswith('.png')])}")
