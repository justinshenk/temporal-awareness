"""Generate 4 diagram options for SPAR midterm report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Shared colors
PROBE = '#4A90D9'
METHOD = '#E67E22'
SAFETY = '#E74C3C'
SCALE = '#2ECC71'
EMERGE = '#9B59B6'
CENTER = '#2C3E50'
BG = '#FAFAFA'


def diagram1_radial():
    """Hub-and-spoke / radial diagram."""
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-7, 7)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Center
    center = plt.Circle((0, 0), 1.3, color=CENTER, zorder=5)
    ax.add_patch(center)
    ax.text(0, 0.15, 'Temporal', ha='center', va='center', fontsize=11, fontweight='bold', color='white', zorder=6)
    ax.text(0, -0.25, 'Representations', ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=6)
    ax.text(0, -0.6, 'in LLMs', ha='center', va='center', fontsize=9, fontweight='bold', color='white', zorder=6)

    themes = [
        {'label': 'Probing &\nSteering', 'color': PROBE, 'angle': 90,
         'fellows': ['Paudel: 99.2%\nprobe, 4× steer', 'Obiso: last 3\nlayers, clock', 'Sharma: cross-\nmodel', 'Mraz: pooled\nprobes']},
        {'label': 'Baselines &\nMethodology', 'color': METHOD, 'angle': 162,
         'fellows': ['Dahiya: baseline\nstaircase', 'No planning in\n1–7B models']},
        {'label': 'Safety\nPhenomena', 'color': SAFETY, 'angle': 234,
         'fellows': ['Bagga: sycophancy\n1.5–15.5%', 'Roy: context\nfatigue', 'Bandyopadhyay:\nuncertainty']},
        {'label': 'Scale &\nStability', 'color': SCALE, 'angle': 306,
         'fellows': ['Molofsky: patience\ndegradation', 'Gupta: phase\ntransitions']},
        {'label': 'Emerging\nDirections', 'color': EMERGE, 'angle': 18,
         'fellows': ['Goel: token\ntrajectories']},
    ]

    for t in themes:
        ang = np.radians(t['angle'])
        r_theme = 3.2
        tx, ty = r_theme * np.cos(ang), r_theme * np.sin(ang)

        # Line from center to theme
        ax.plot([0, tx], [0, ty], color=t['color'], linewidth=3, zorder=2, alpha=0.6)

        # Theme circle
        circ = plt.Circle((tx, ty), 0.95, color=t['color'], zorder=4)
        ax.add_patch(circ)
        ax.text(tx, ty, t['label'], ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=5)

        # Fellow nodes
        n = len(t['fellows'])
        spread = 30
        base_ang = t['angle']
        for i, f in enumerate(t['fellows']):
            offset = (i - (n - 1) / 2) * spread
            fang = np.radians(base_ang + offset)
            r_fellow = 5.5
            fx, fy = r_fellow * np.cos(fang), r_fellow * np.sin(fang)

            ax.plot([tx, fx], [ty, fy], color=t['color'], linewidth=1.2, zorder=1, alpha=0.4)
            box = mpatches.FancyBboxPatch((fx - 0.95, fy - 0.45), 1.9, 0.9,
                                          boxstyle='round,pad=0.1', facecolor='white',
                                          edgecolor=t['color'], linewidth=1.2, zorder=3)
            ax.add_patch(box)
            ax.text(fx, fy, f, ha='center', va='center', fontsize=5.5, zorder=4)

    fig.suptitle('Option 1: Radial / Hub-and-Spoke', fontsize=14, fontweight='bold', y=0.97)
    plt.tight_layout()
    fig.savefig('diagram1-radial.png', dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()


def diagram2_pipeline():
    """Left-to-right pipeline diagram."""
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-0.5, 16)
    ax.set_ylim(-8.5, 2.5)
    ax.axis('off')

    stages = [
        {'label': 'Data &\nDatasets', 'color': PROBE, 'x': 1.5,
         'fellows': [('Dahiya', '500 code samples'), ('Paudel', '1000 implicit pairs'), ('Bagga', 'sycophancy bench'), ('Molofsky', '8×4×10 reps')]},
        {'label': 'Linear\nProbes', 'color': METHOD, 'x': 5.5,
         'fellows': [('Sharma', 'cross-model probes'), ('Mraz', 'pooled activation'), ('Bandyopadhyay', 'uncertainty probes'), ('Goel', 'token trajectories')]},
        {'label': 'Activation\nSteering', 'color': SCALE, 'x': 9.5,
         'fellows': [('Paudel', '4× effect L22'), ('Obiso', 'causal L29–31'), ('Gupta', 'phase transitions')]},
        {'label': 'Safety\nEvaluation', 'color': SAFETY, 'x': 13.5,
         'fellows': [('Obiso', 'steering → harm'), ('Bagga', 'sycophancy 1.5–15.5%'), ('Roy', 'entropy collapse'), ('Dahiya', 'baseline staircase')]},
    ]

    for i, s in enumerate(stages):
        # Stage box
        box = mpatches.FancyBboxPatch((s['x'] - 1.3, 0.3), 2.6, 1.5,
                                      boxstyle='round,pad=0.15', facecolor=s['color'],
                                      edgecolor='none', zorder=3)
        ax.add_patch(box)
        ax.text(s['x'], 1.05, s['label'], ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=4)

        # Arrow to next
        if i < len(stages) - 1:
            nx = stages[i + 1]['x']
            ax.annotate('', xy=(nx - 1.5, 1.05), xytext=(s['x'] + 1.5, 1.05),
                        arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=2.5))

        # Fellow boxes below
        for j, (name, desc) in enumerate(s['fellows']):
            y = -0.8 - j * 1.6
            fbox = mpatches.FancyBboxPatch((s['x'] - 1.15, y - 0.55), 2.3, 1.1,
                                           boxstyle='round,pad=0.1', facecolor='white',
                                           edgecolor='#BDC3C7', linewidth=1, zorder=2)
            ax.add_patch(fbox)
            ax.text(s['x'], y + 0.1, name, ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)
            ax.text(s['x'], y - 0.2, desc, ha='center', va='center', fontsize=6.5, color='#555', zorder=3)
            ax.plot([s['x'], s['x']], [0.3, y + 0.55], color='#BDC3C7', linewidth=0.8, zorder=1)

    fig.suptitle('Option 2: Pipeline', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig('diagram2-pipeline.png', dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()


def diagram3_matrix():
    """Model × Research Question coverage matrix."""
    models = ['GPT-2\n(124M–1.5B)', 'Pythia\n(410M–2.8B)', 'SantaCoder', 'CodeLlama\n7B', 'Llama-3.2\n(1B, 3B)', 'Qwen3\n(4B–30B)', 'Phi-3\nmini']
    questions = ['Temporal\nHorizon', 'Lookahead\nPlanning', 'Context\nFatigue', 'Sycophancy', 'Uncertainty', 'Patience\nDegradation', 'Token\nTrajectories']

    # 0=not tested, 1=confirmed, 2=null, 3=in progress
    data = np.zeros((len(models), len(questions)))
    names = [['' for _ in questions] for _ in models]

    # Temporal Horizon
    names[4][0] = 'Paudel'; data[4][0] = 1
    names[5][0] = 'Obiso\nSharma'; data[5][0] = 1
    names[6][0] = 'Sharma'; data[6][0] = 1
    names[0][0] = 'Mraz'; data[0][0] = 3

    # Lookahead Planning (null across all tested)
    for i in range(5):
        names[i][1] = 'Dahiya'; data[i][1] = 2

    # Context Fatigue
    names[5][2] = 'Roy'; data[5][2] = 3

    # Sycophancy
    names[5][3] = 'Bagga'; data[5][3] = 3

    # Uncertainty
    names[5][4] = 'Bandyo.'; data[5][4] = 1

    # Patience Degradation
    names[4][5] = 'Molofsky'; data[4][5] = 1
    names[5][5] = 'Molofsky'; data[5][5] = 1

    # Token Trajectories
    names[0][6] = 'Goel'; data[0][6] = 3

    colors = {0: '#ECF0F1', 1: '#A9DFBF', 2: '#F5B7B1', 3: '#FAD7A0'}

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
    ax.set_facecolor(BG)

    cell_w, cell_h = 1.6, 0.85
    x0, y0 = 2.5, 0.5

    # Column headers
    for j, q in enumerate(questions):
        ax.text(x0 + j * cell_w + cell_w / 2, y0 + len(models) * cell_h + 0.4, q,
                ha='center', va='center', fontsize=7, fontweight='bold', color=CENTER)

    # Row headers
    for i, m in enumerate(models):
        yi = y0 + (len(models) - 1 - i) * cell_h
        ax.text(x0 - 0.2, yi + cell_h / 2, m, ha='right', va='center', fontsize=7, fontweight='bold', color=CENTER)

    # Cells
    for i in range(len(models)):
        for j in range(len(questions)):
            yi = y0 + (len(models) - 1 - i) * cell_h
            xj = x0 + j * cell_w
            c = colors[data[i][j]]
            rect = mpatches.FancyBboxPatch((xj + 0.05, yi + 0.05), cell_w - 0.1, cell_h - 0.1,
                                           boxstyle='round,pad=0.05', facecolor=c,
                                           edgecolor='white', linewidth=1.5, zorder=2)
            ax.add_patch(rect)
            label = names[i][j] if names[i][j] else '–'
            ax.text(xj + cell_w / 2, yi + cell_h / 2, label,
                    ha='center', va='center', fontsize=5.5, color='#333' if names[i][j] else '#AAA', zorder=3)

    # Legend
    ly = y0 - 1.0
    for val, label, color in [(1, 'Confirmed signal', '#A9DFBF'), (2, 'Null result', '#F5B7B1'), (3, 'In progress', '#FAD7A0')]:
        lx = x0 + 2.5 + val * 3.5
        rect = mpatches.FancyBboxPatch((lx, ly), 0.5, 0.35, boxstyle='round,pad=0.05',
                                       facecolor=color, edgecolor='none', zorder=2)
        ax.add_patch(rect)
        ax.text(lx + 0.7, ly + 0.17, label, ha='left', va='center', fontsize=7, color='#333')

    ax.set_xlim(-0.5, x0 + len(questions) * cell_w + 0.5)
    ax.set_ylim(ly - 0.5, y0 + len(models) * cell_h + 1.2)
    ax.axis('off')

    fig.suptitle('Option 3: Model × Research Question Matrix', fontsize=14, fontweight='bold', y=0.97)
    plt.tight_layout()
    fig.savefig('diagram3-matrix.png', dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()


def diagram4_stack():
    """Conceptual stack / layer diagram."""
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-2, 15)
    ax.set_ylim(-0.5, 12)
    ax.axis('off')

    layer_w = 13.5
    layer_x = 0.5

    # --- Bottom layer: Models (simplified text line) ---
    my = 0.5
    mh = 1.8
    rect = mpatches.FancyBboxPatch((layer_x, my), layer_w, mh,
                                   boxstyle='round,pad=0.15', facecolor='#EBF5FB',
                                   edgecolor=PROBE, linewidth=2.5, zorder=1)
    ax.add_patch(rect)
    ax.text(layer_x + layer_w / 2, my + mh * 0.62, 'Models (1B – 30B)',
            fontsize=16, fontweight='bold', color=PROBE, ha='center', va='center', zorder=3)
    ax.text(layer_x + layer_w / 2, my + mh * 0.22,
            'GPT-2  ·  Pythia  ·  SantaCoder  ·  CodeLlama  ·  Llama-3.2  ·  Qwen3  ·  Phi-3',
            fontsize=11, color='#555', ha='center', va='center', zorder=3)

    # --- Middle and top layers with fellow tags ---
    tag_layers = [
        {'label': 'Probing & Steering (Layers 19–31)', 'color': METHOD, 'y': 3.1, 'h': 3.6, 'bg': '#FDF2E9',
         'tags': [
             ('99.2% probe\naccuracy L26', 'Paudel'),
             ('Causal patching\nL29–31', 'Obiso'),
             ('Cross-model\ncomparison', 'Sharma'),
             ('Pooled-activation\nprobes', 'Mraz'),
             ('Baseline staircase\nprotocol', 'Dahiya'),
         ],
         'annotation': 'Early layers: weak signal ————→ Late layers: strong signal'},
        {'label': 'Safety Applications', 'color': SAFETY, 'y': 7.5, 'h': 3.6, 'bg': '#FDEDEC',
         'tags': [
             ('Steering →\nsafety / harm', 'Obiso'),
             ('Sycophancy\ndetection', 'Bagga'),
             ('Context\nfatigue', 'Roy'),
             ('Uncertainty\nencoding', 'Bandyopadhyay'),
             ('Patience\ndegradation', 'Molofsky'),
         ],
         'annotation': 'Oversight  ·  Deception Detection  ·  Runtime Monitoring'},
    ]

    for layer in tag_layers:
        y = layer['y']
        h = layer['h']
        rect = mpatches.FancyBboxPatch((layer_x, y), layer_w, h,
                                       boxstyle='round,pad=0.15', facecolor=layer['bg'],
                                       edgecolor=layer['color'], linewidth=2.5, zorder=1)
        ax.add_patch(rect)

        ax.text(layer_x + 0.8, y + h - 0.35, layer['label'], fontsize=15, fontweight='bold',
                color=layer['color'], zorder=3)

        n = len(layer['tags'])
        tag_w = 2.3
        gap = 0.25
        total_w = n * tag_w + (n - 1) * gap
        start_x = layer_x + (layer_w - total_w) / 2
        for i, (desc, name) in enumerate(layer['tags']):
            tx = start_x + i * (tag_w + gap)
            tag_h = 2.0
            tag_y = y + 0.5
            tbox = mpatches.FancyBboxPatch((tx, tag_y), tag_w, tag_h,
                                           boxstyle='round,pad=0.12', facecolor='white',
                                           edgecolor='#BDC3C7', linewidth=1.2, zorder=2)
            ax.add_patch(tbox)
            ax.text(tx + tag_w / 2, tag_y + tag_h * 0.62, desc,
                    ha='center', va='center', fontsize=11, fontweight='bold', zorder=3)
            ax.text(tx + tag_w / 2, tag_y + tag_h * 0.2, f'({name})',
                    ha='center', va='center', fontsize=10, color='#666', zorder=3)

        ax.text(layer_x + layer_w / 2, y + 0.2, layer['annotation'], ha='center', va='center',
                fontsize=10, color='#999', style='italic', zorder=3)

    # Vertical arrows
    for x in [4.5, 7.25, 10]:
        ax.annotate('', xy=(x, 3.1), xytext=(x, 2.3),
                    arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=2.5))
        ax.annotate('', xy=(x, 7.5), xytext=(x, 6.7),
                    arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=2.5))

    # Side label
    ax.text(-1.0, 6, 'Increasing\nAbstraction', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#7F8C8D', rotation=90)
    ax.annotate('', xy=(-1.0, 10.5), xytext=(-1.0, 1.5),
                arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=2.5))

    fig.suptitle('Research Structure: Temporal Activation Monitors', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig('diagram4-stack.png', dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()


if __name__ == '__main__':
    diagram1_radial()
    diagram2_pipeline()
    diagram3_matrix()
    diagram4_stack()
    print('All 4 diagrams generated.')
