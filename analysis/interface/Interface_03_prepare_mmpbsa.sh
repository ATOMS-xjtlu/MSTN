#!/bin/bash
# ============================================================
# MM-GBSA topology preparation: build complex/receptor/ligand prmtop files and fix IFBOX
# ============================================================
# Three topologies:
#   complex.prmtop  = ActRII(1-109) + MSTN(314-408), 204 residues
#   receptor.prmtop = ActRII(1-109), 109 residues
#   ligand.prmtop   = MSTN(314-408), 95 residues
#
# Important: IFBOX is the 28th POINTERS value in the Amber prmtop (0-based index 27).
#       After cpptraj parmstrip IFBOX remains 1, whereas GB calculations require 0, so it must be fixed manually.
# ============================================================
set -e
WORK=/home/p2600254/interface_analysis
CPPTRAJ=/home/p2600254/miniconda3/envs/md/bin/cpptraj
TOP=/mnt/c/Users/p2600254/Desktop/project/supplementary/interface_analysis/main/main/complex-amber.top
cd "$WORK"

# 1) Generate the three topologies with cpptraj parmstrip
gen() {
  local name=$1 mask=$2
  $CPPTRAJ -p "$TOP" > /dev/null 2>&1 <<EOF
parmstrip $mask
parmwrite out $name.prmtop
EOF
}

gen complex  '!:1-109,314-408'
gen receptor '!:1-109'
gen ligand   '!:314-408'

# 2) Set IFBOX = 0 in Python while keeping the fixed-width I8 format
python3 <<'PYEOF'
import re
for name in ['complex', 'receptor', 'ligand']:
    path = f'/home/p2600254/interface_analysis/{name}.prmtop'
    with open(path) as f:
        content = f.read()
    m = re.search(r'%FLAG POINTERS\s*\n%FORMAT\(10I8\)\s*\n(.*?)(?=\n%FLAG|\Z)', content, re.DOTALL)
    ints = m.group(1).split()
    assert len(ints) >= 28, f"{name}: not enough POINTERS values"
    ints[27] = '0'   # IFBOX = 0
    newsec = '\n'.join(''.join(f'{int(v):8d}' for v in ints[i:i+10]) for i in range(0, len(ints), 10)) + '\n'
    content = content[:m.start(1)] + newsec + content[m.end(1):]
    with open(path, 'w') as f:
        f.write(content)
    print(f"{name}.prmtop: IFBOX set to 0")
PYEOF

echo "=== Generation complete ==="
ls -la complex.prmtop receptor.prmtop ligand.prmtop
