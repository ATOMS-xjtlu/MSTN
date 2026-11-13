import os, shutil
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
BASE = r"response_letter\polished_CN\11.9_revision"
SI = BASE + r"\SI_revised_11.9.docx"
R1 = BASE + r"\Response to Reviewer 1_11.9.docx"
BK = os.path.join(BASE, "_backup")
for p in (SI, R1):
    d = os.path.join(BK, os.path.basename(p) + ".pre_figS19")
    if not os.path.exists(d):
        shutil.copy2(p, d)

CAPTION = ("Figure S19. Within-cluster distributions of the three collective variables for the eleven states. Frames of every cMD "
           "trajectory were assigned to the cluster whose centroid is nearest in the standardised collective-variable space, the "
           "centroids being computed from frames sampled every 5 ns; histograms are normalised to unit area. The red dashed line "
           "marks the value of the representative frame of that state in Table S3, and the cluster index and the number of assigned "
           "frames are given for each state.")

doc = Document(SI)
# locate the Figure S18 caption paragraph to inherit formatting
cap18 = None
for p in doc.element.body.iter(W + "p"):
    t = "".join(x.text or "" for x in p.iter(W + "t"))
    if t.startswith("Figure S18."):
        cap18 = p
if cap18 is None:
    raise SystemExit("S18 caption not found")

import copy
cap_el = copy.deepcopy(cap18)
cap18.addnext(cap_el)
from docx.text.paragraph import Paragraph
cap = Paragraph(cap_el, doc)
# rewrite text of the copied caption
runs = cap.runs
if runs:
    runs[0].text = CAPTION
    for r in runs[1:]:
        r.text = ""
else:
    cap.add_run(CAPTION)

img_el = copy.deepcopy(cap_el)
cap_el.addprevious(img_el)
img_p = Paragraph(img_el, doc)
for r in list(img_p.runs):
    r._element.getparent().remove(r._element)
img_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
img_p.add_run().add_picture(r"_tmp3\Fig_S19_cluster_distributions.png", width=Inches(6.0))
doc.save(SI)
print("SI: figure + caption inserted")

doc = Document(SI)
paras = ["".join(x.text or "" for x in p.iter(W + "t")) for p in doc.element.body.iter(W + "p")]
tail = [t for t in paras if t.strip()][-2:]
for t in tail:
    print("  tail:", t[:120])

# letter reference
OLD = ("and Figure S10 (distance to the 25th nearest neighbour, used to choose the RMSD cut-off).")
NEW = ("Figure S10 (distance to the 25th nearest neighbour, used to choose the RMSD cut-off) and Figure S19 (within-cluster "
       "distributions of the three collective variables for the eleven states, with the representative frame marked).")
doc = Document(R1)
n = 0
for p in doc.element.body.iter(W + "p"):
    for r in p.iter(W + "t"):
        if r.text and OLD in r.text:
            r.text = r.text.replace(OLD, NEW, 1); n += 1
            break
doc.save(R1)
print("letter refs updated:", n)
