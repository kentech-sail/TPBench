"""Draw ranks from the manuscript's saved aggregate values, without inference."""
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

import argparse, hashlib, statistics
parser = argparse.ArgumentParser(description='Draw corrected ranks from the released exact means.')
parser.add_argument('--summary', type=Path, required=True)
parser.add_argument('--out', type=Path, default=Path(__file__).with_name('tpbench_probe_ranks.pdf'))
args = parser.parse_args()
summary = json.loads(args.summary.read_text())
data = {'p1_means': {}, 'p2': {}}
for dataset, p2key in [('sgd', 'sgd_r30'), ('multiwoz', 'mw_r30')]:
    data['p1_means'][dataset] = {method: {'strict': row['p1_em_strict']['mean'], 'loose': row['p1_em_loose']['mean']} for method, row in summary['p1'][dataset+'_r30'].items()}
    data['p2'][dataset] = {method: row['p3_em_strict']['mean'] for method, row in summary['p3'][p2key].items()}

methods = ['recency','random_seed42','first_n','uniform_stride','attention_h2o_cache','embedding_mmr_cache','llmlingua2_cache']
names = ['Recency','Random selection','Earliest turns','Uniform stride','H2O proxy','Embedding MMR','LLMLingua-2']
colors = ['#777777','#0072B2','#D55E00','#009E73','#CC79A7','#56B4E9','#AA8800']
def ranks(values):
    # Round only floating-point noise; equal empirical scores share midrank.
    values = [round(v,12) for v in values]
    return [1+sum(w>v for w in values)+(sum(w==v for w in values)-1)/2 for v in values]

out = args.out
rank_audit = {'summary_path': str(args.summary.resolve().relative_to(args.summary.resolve().parents[2])), 'summary_sha256': hashlib.sha256(args.summary.read_bytes()).hexdigest(), 'panels': {}}
c = canvas.Canvas(str(out),pagesize=(640,310),invariant=1)
c.setTitle('TPBench: initial-goal and current-value method ranks')
c.setAuthor('')
for panel,dataset in enumerate(['sgd','multiwoz']):
    left=42+panel*320
    xs=[left+28,left+130,left+232]
    cols=[ranks([data['p1_means'][dataset][m][metric] for m in methods]) for metric in ['strict','loose']]
    cols.append(ranks([data['p2'][dataset][m] for m in methods]))
    rank_audit['panels'][dataset] = {'methods': methods, 'values': [[data['p1_means'][dataset][m][metric] for m in methods] for metric in ['strict', 'loose']] + [[data['p2'][dataset][m] for m in methods]], 'ranks': cols, 'spearman_p1strict_p2strict': statistics.correlation(cols[0], cols[2])}
    c.setFillColor(HexColor('#222222'));c.setFont('Helvetica-Bold',16)
    c.drawCentredString(left+130,292, 'SGD' if panel==0 else 'MultiWOZ')
    for r in range(1,8):
        y=264-(r-1)*27
        c.setStrokeColor(HexColor('#dddddd'));c.setLineWidth(.5);c.line(xs[0],y,xs[-1],y)
        c.setFont('Helvetica',12);c.drawRightString(left+14,y-3,str(r))
    for i,m in enumerate(methods):
        c.setStrokeColor(HexColor(colors[i]));c.setFillColor(HexColor(colors[i]));c.setLineWidth(1.7)
        ys=[264-(col[i]-1)*27 for col in cols]
        c.setDash([4,2] if i in [2,4,6] else [])
        for j in range(2):c.line(xs[j],ys[j],xs[j+1],ys[j+1])
        c.setDash([])
        for x,y in zip(xs,ys):
            if i%2:c.rect(x-2.5,y-2.5,5,5,stroke=1,fill=0)
            else:c.circle(x,y,2.7,stroke=1,fill=0)
    c.setFillColor(HexColor('#222222'));c.setFont('Helvetica',13)
    for x,s in zip(xs,['P1 strict','P1 loose','P2 strict']):c.drawCentredString(x,83,s)
c.setFont('Helvetica',12);c.drawString(16,292,'Rank')
c.setFont('Helvetica',13)
for i,(name,color) in enumerate(zip(names,colors)):
    x=20+(i%4)*157;y=48-(i//4)*19
    c.setStrokeColor(HexColor(color));c.setLineWidth(2);c.line(x,y+3,x+18,y+3)
    c.setFillColor(HexColor('#222222'));c.drawString(x+24,y,name)
c.setFont('Helvetica',12);c.drawCentredString(320,8,'Rank 1 is best; ties use average ranks. Retained turn fraction r = 0.30.')
c.save()
out.with_suffix('.inputs.json').write_text(json.dumps(rank_audit, indent=2)+'\n')
print(out)
