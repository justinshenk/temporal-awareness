import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size':11,'font.family':'sans-serif','axes.titlesize':13,'axes.labelsize':12,'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight'})
FIGDIR = "results/lookahead/final/figures"
os.makedirs(FIGDIR, exist_ok=True)

# Fig 1: Baseline staircase
models = [("GPT2S",80.3,84.6),("GPT2M",78.7,87.6),("GPT2XL",80.7,87.4),("Py410",80.9,89.2),("Py1B",82.3,90.1),("Py1.4B",83.2,90.0),("Py2.8B",83.2,91.1),("Santa",91.1,93.1),("CdLlama",88.6,94.5),("Llama1B",86.6,93.7),("Llama1BI",83.6,94.3)]
x=np.arange(len(models)); w=0.35
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,7),height_ratios=[3,1],gridspec_kw={'hspace':0.3})
ax1.bar(x-w/2,[m[1] for m in models],w,label='Probe',color='#4C72B0',alpha=.85)
ax1.bar(x+w/2,[m[2] for m in models],w,label='Name+Params',color='#DD8452',alpha=.85)
ax1.set_ylabel('Accuracy (%)'); ax1.set_title('Baseline Staircase: Name+Params Beats Probe Everywhere')
ax1.set_xticks(x); ax1.set_xticklabels([m[0] for m in models],rotation=45,ha='right'); ax1.legend(); ax1.set_ylim(70,100); ax1.grid(axis='y',alpha=.3)
gaps=[m[1]-m[2] for m in models]
ax2.bar(x,gaps,0.6,color=['#C44E52' if g<-5 else '#CCB974' for g in gaps],alpha=.85)
ax2.set_ylabel('Gap (%)'); ax2.set_title('Gap (Probe − Name+Params): All Negative'); ax2.set_xticks(x); ax2.set_xticklabels([m[0] for m in models],rotation=45,ha='right'); ax2.axhline(0,color='k',lw=.8); ax2.set_ylim(-15,2); ax2.grid(axis='y',alpha=.3)
plt.savefig(f"{FIGDIR}/fig1_baseline_staircase.png"); plt.savefig(f"{FIGDIR}/fig1_baseline_staircase.pdf"); plt.close(); print("  Fig 1 ✓")

# Fig 2: Future Lens K decay
fig,ax=plt.subplots(figsize=(8,5))
for name,gaps,c,m in [('Pythia-2.8B',[18.7,10.3,6.0,-1.2],'#4C72B0','o'),('Qwen-1.5B',[51.3,24.5,11.1,2.7],'#DD8452','s'),('GPT-J-6B',[30.9,15.8,5.2,4.9],'#55A868','^'),('Qwen-7B',[60.3,22.5,10.6,5.0],'#C44E52','D')]:
    ax.plot([1,2,3,5],gaps,'-'+m,color=c,label=name,lw=2,ms=8)
ax.axhline(0,color='gray',ls='--',alpha=.5); ax.fill_between([.5,5.5],-5,5,alpha=.1,color='gray')
ax.set_xlabel('Prediction Distance (K)'); ax.set_ylabel('Gap: Probe − Context Emb (%)'); ax.set_title('Future Lens K Decay: Universal Across Architectures'); ax.set_xticks([1,2,3,5]); ax.legend(); ax.grid(alpha=.3); ax.set_xlim(.5,5.5)
plt.savefig(f"{FIGDIR}/fig2_future_lens_k_decay.png"); plt.savefig(f"{FIGDIR}/fig2_future_lens_k_decay.pdf"); plt.close(); print("  Fig 2 ✓")

# Fig 3: Intermediate domains
domains=['Poetry','CoT Math','Code','Free Prose','Struct Prose']; k1=[31.9,30.3,22.5,21.7,21.3]; k3=[-1.2,28.0,11.0,18.9,11.0]
x=np.arange(len(domains)); w=.35
fig,ax=plt.subplots(figsize=(10,5))
ax.bar(x-w/2,k1,w,label='K=1',color='#4C72B0',alpha=.85); ax.bar(x+w/2,k3,w,label='K=3',color='#DD8452',alpha=.85)
ax.set_ylabel('Gap: Probe − Context Emb (%)'); ax.set_title('Intermediate Domains: CoT Maintains Signal, Poetry Collapses'); ax.set_xticks(x); ax.set_xticklabels(domains); ax.legend(); ax.axhline(0,color='k',lw=.8); ax.grid(axis='y',alpha=.3)
ax.annotate('Collapses\nat K=3',xy=(0,-1.2),xytext=(0,-8),fontsize=9,ha='center',color='red',arrowprops=dict(arrowstyle='->',color='red'))
ax.annotate('Maintains\n+28%',xy=(1,28),xytext=(1.5,33),fontsize=9,ha='center',color='green',arrowprops=dict(arrowstyle='->',color='green'))
plt.savefig(f"{FIGDIR}/fig3_intermediate_domains.png"); plt.savefig(f"{FIGDIR}/fig3_intermediate_domains.pdf"); plt.close(); print("  Fig 3 ✓")

# Fig 4: Generation-time decay
fig,ax=plt.subplots(figsize=(8,5))
steps=list(range(21)); cl=[95.3,93.1,90.2,87.5,84.3,81,78.2,75.5,73.1,71,69.2,67.5,66.1,65,64.1,63.5,63.2,63,62.9,62.9,62.9]
py=[83.2,81,79.5,78.1,76.8,75.5,74.5,73.8,73.2,72.8,72.5,72.3,72.1,72,71.9,71.8,71.7,71.7,71.6,71.6,71.6]
ax.plot(steps,cl,'-o',color='#C44E52',label='CodeLlama-7B',lw=2,ms=4); ax.plot(steps,py,'-s',color='#4C72B0',label='Pythia-2.8B',lw=2,ms=4)
ax.set_xlabel('Generation Step'); ax.set_ylabel('Probe Accuracy (%)'); ax.set_title('Generation-Time Decay: Models Lose Info During Generation'); ax.legend(); ax.grid(alpha=.3)
plt.savefig(f"{FIGDIR}/fig4_gentime_decay.png"); plt.savefig(f"{FIGDIR}/fig4_gentime_decay.pdf"); plt.close(); print("  Fig 4 ✓")

# Fig 5: Fixed positions
pos=['BOS\n(pos=0)','def\n(pos=1)','name\n(pos=2)','paren\n(pos=3)','params\n(pos=5)','last\nposition']; accs=[22,22,81,80,80,94]
colors=['#AAA','#AAA','#4C72B0','#6C92D0','#6C92D0','#DD8452']
fig,ax=plt.subplots(figsize=(8,5))
bars=ax.bar(pos,accs,color=colors,alpha=.85,edgecolor='k',lw=.5); ax.axhline(20,color='red',ls='--',alpha=.5,label='Chance (20%)')
ax.set_ylabel('Probe Accuracy (%)'); ax.set_title('Fixed-Position Probing (100 Sigs): Info Lives in Function Name'); ax.legend(); ax.set_ylim(0,100); ax.grid(axis='y',alpha=.3)
for b,a in zip(bars,accs): ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{a}%',ha='center',va='bottom',fontsize=10,fontweight='bold')
plt.savefig(f"{FIGDIR}/fig5_fixed_positions.png"); plt.savefig(f"{FIGDIR}/fig5_fixed_positions.pdf"); plt.close(); print("  Fig 5 ✓")

# Fig 6: Spearman
beh=[0,0,0,26.7,32,38.7,32.7,41.3,42.7,34,34]; probe=[80.3,78.7,80.7,80.9,82.3,83.2,83.2,91.1,88.6,86.6,83.6]; gap=[-3.9,-8.9,-5.5,-8.5,-7.8,-6.8,-7.3,-2,-5.9,-7.1,-10.7]
names=['GPT2S','GPT2M','GPT2XL','Py410','Py1B','Py1.4B','Py2.8B','Santa','CdLl','Llama','LlI']
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5))
a1.scatter(beh,probe,c='#4C72B0',s=80,zorder=5)
for i,n in enumerate(names): a1.annotate(n,(beh[i],probe[i]),fontsize=7,xytext=(5,5),textcoords='offset points')
z=np.polyfit(beh,probe,1); a1.plot(np.linspace(-5,50),np.poly1d(z)(np.linspace(-5,50)),'--',color='#4C72B0',alpha=.5)
a1.set_xlabel('Behavioral Acc (%)'); a1.set_ylabel('Best Probe Acc (%)'); a1.set_title('ρ = +0.940 (p < 0.001)\nCapability → Probe Signal'); a1.grid(alpha=.3)
a2.scatter(beh,gap,c='#C44E52',s=80,zorder=5)
for i,n in enumerate(names): a2.annotate(n,(beh[i],gap[i]),fontsize=7,xytext=(5,5),textcoords='offset points')
a2.axhline(0,color='k',lw=.5); a2.set_xlabel('Behavioral Acc (%)'); a2.set_ylabel('Gap: Probe − N+P (%)'); a2.set_title('ρ = −0.078 (p = 0.819)\nCapability ≠ Planning'); a2.grid(alpha=.3)
plt.suptitle('Capability Predicts Probe Signal, Not Planning',fontsize=14,y=1.02); plt.tight_layout()
plt.savefig(f"{FIGDIR}/fig6_spearman.png"); plt.savefig(f"{FIGDIR}/fig6_spearman.pdf"); plt.close(); print("  Fig 6 ✓")

# Fig 7: Misleading names
mdls=['GPT-2 XL','Pythia-2.8B','CodeLlama-7B']; fp=[56,56,56]; fn=[22,28,34]; fne=[22,16,10]
x=np.arange(3); w=.25
fig,ax=plt.subplots(figsize=(8,5))
ax.bar(x-w,fp,w,label='Follows params',color='#4C72B0',alpha=.85); ax.bar(x,fn,w,label='Follows name',color='#DD8452',alpha=.85); ax.bar(x+w,fne,w,label='Neither',color='#AAA',alpha=.85)
ax.set_ylabel('% of Examples'); ax.set_title('Misleading Names: Models Follow Parameters, Not Names'); ax.set_xticks(x); ax.set_xticklabels(mdls); ax.legend(); ax.grid(axis='y',alpha=.3)
plt.savefig(f"{FIGDIR}/fig7_misleading_names.png"); plt.savefig(f"{FIGDIR}/fig7_misleading_names.pdf"); plt.close(); print("  Fig 7 ✓")

# Fig 8: Pal comparison
ks=['K=1','K=3','K=5']; x=np.arange(3); w=.25
fig,ax=plt.subplots(figsize=(8,5))
ax.bar(x-w,[4.8,4.6,4.1],w,label='Pal et al. (PCA)',color='#AAA',alpha=.85); ax.bar(x,[2.0,2.7,2.5],w,label='Pal et al. (full)',color='#CCB974',alpha=.85); ax.bar(x+w,[85.4,44.4,34.0],w,label='Our probe',color='#4C72B0',alpha=.85)
ax.set_ylabel('Accuracy (%)'); ax.set_title('Classification Probes vs Linear Hidden-State Mapping'); ax.set_xticks(x); ax.set_xticklabels(ks); ax.legend(); ax.grid(axis='y',alpha=.3)
plt.savefig(f"{FIGDIR}/fig8_pal_comparison.png"); plt.savefig(f"{FIGDIR}/fig8_pal_comparison.pdf"); plt.close(); print("  Fig 8 ✓")

print(f"\nAll 8 figures saved to {FIGDIR}/")

# Fig 9: Scrambled CoT ablation
def fig9_scrambled_cot():
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
    x=np.arange(2); w=0.3
    ax1.bar(x-w/2,[31.1,30.2],w,label='K=1',color='#4C72B0',alpha=.85)
    ax1.bar(x+w/2,[28.8,29.2],w,label='K=3',color='#DD8452',alpha=.85)
    ax1.set_ylabel('Gap (%)'); ax1.set_title('GPT-J-6B: Gap Survives Scrambling')
    ax1.set_xticks(x); ax1.set_xticklabels(['CoT (normal)','CoT (scrambled)']); ax1.legend(); ax1.grid(axis='y',alpha=.3); ax1.set_ylim(0,40)
    ax1.annotate('Nearly\nidentical',xy=(1.15,29),fontsize=10,color='green',fontweight='bold')
    ax2.bar(x-w/2,[66.2,53.5],w,label='K=1',color='#4C72B0',alpha=.85)
    ax2.bar(x+w/2,[33.4,20.8],w,label='K=3',color='#DD8452',alpha=.85)
    ax2.set_ylabel('Gap (%)'); ax2.set_title('Qwen-7B: Partial Reduction, Still Large')
    ax2.set_xticks(x); ax2.set_xticklabels(['CoT (normal)','CoT (scrambled)']); ax2.legend(); ax2.grid(axis='y',alpha=.3); ax2.set_ylim(0,75)
    ax2.annotate('Still +21%\nat K=3',xy=(1.15,21),fontsize=10,color='green',fontweight='bold')
    plt.suptitle('Scrambled CoT: Signal Is NOT Template Following',fontsize=14,y=1.02); plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig9_scrambled_cot.png"); plt.savefig(f"{FIGDIR}/fig9_scrambled_cot.pdf"); plt.close()
    print("  Fig 9: Scrambled CoT ✓")

fig9_scrambled_cot()
