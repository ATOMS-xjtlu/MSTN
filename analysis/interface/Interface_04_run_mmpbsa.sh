#!/bin/bash
# ============================================================
# Run the MM-GBSA binding free-energy calculation
# ============================================================
# Input: complex.prmtop / receptor.prmtop / ligand.prmtop + traj_*.mdcrd
# Output: FINAL_1us.dat / FINAL_2us.dat (with energy decomposition)
#
# Note: the vdW component is unreliable because of an atom-type reindexing bug in cpptraj parmstrip;
#     the electrostatic (EEL), polar solvation (EGB) and non-polar (ESURF) components are reliable.
# ============================================================
set -e
cd /home/p2600254/interface_analysis
export AMBERHOME=/home/p2600254/miniconda3/envs/md
MMPBSA=/home/p2600254/miniconda3/envs/md/bin/MMPBSA.py

# MMPBSA input (GB model igb=2, 0.15 M salt)
cat > mmpbsa.in <<'EOF'
&general
  startframe=1, endframe=100, interval=1,
  verbose=2,
/
&gb
  igb=2, saltcon=0.150,
/
EOF

for tag in 1us 2us; do
  echo "========== MMPBSA.py ($tag) =========="
  $MMPBSA -O -i mmpbsa.in -o FINAL_${tag}.dat \
    -cp complex.prmtop -rp receptor.prmtop -lp ligand.prmtop -y traj_${tag}.mdcrd 2>&1 | tail -6
  echo "--- ${tag} DELTA TOTAL ---"
  grep 'DELTA TOTAL' FINAL_${tag}.dat
  echo ""
done
