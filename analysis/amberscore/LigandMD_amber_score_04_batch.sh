#!/bin/bash
# Batch Amber Score (8 systems), run on the DGX
# Workflow: copy mol2 -> ambpdb -> prepare_amber.pl (-i -c) -> amberize_ligand (-c gas) -> amberize_complex -> dock6
D=/home/yangpeirun_cluster/dock6/dock6
MDIR=/home/yangpeirun_cluster/docking/md_prep
ASDIR=/home/yangpeirun_cluster/docking/amber_score/systems
export DOCK_PARAMETERS=$D/parameters
export AMBERHOME=/home/liuxiaomeng_cluster/software/amber20/amber20_src
export PATH=$AMBERHOME/bin:$D/bin:$PATH

SYSTEMS="state8_CNP0022627_1 state8_CNP0183427_0 state8_CNP0604857_3 state8_CNP0606043_1 state12_CNP0022627_1 state12_CNP0183427_0 state12_CNP0604857_3 state12_CNP0606043_1"

for sys in $SYSTEMS; do
  rec=${sys%%_*}; lig=${sys#*_}
  d=$ASDIR/$sys
  cd "$d" || { echo "Skipping $sys (directory missing)"; continue; }
  echo ""
  echo "================================================================"
  echo "  Amber Score: $sys  (rec=$rec, lig=$lig)"
  echo "================================================================"

  # 1. Copy the all-atom mol2 (without .amber_score in the name to avoid self-truncation)
  cp -f $MDIR/ligands/${sys}_gaff.mol2 ./${lig}.mol2 || { echo "  failed to copy mol2"; continue; }

  # 2. Receptor PDB: must use the original docking receptor so that ligand and receptor share the same coordinate frame
  #    (do not use the receptor extracted with ambpdb from the MD system; it was translated by tleap and the ligand would drift outside)
  cp -f /home/yangpeirun_cluster/docking/original_receptors/${rec}.pdb ./${rec}.pdb
  [ -s ${rec}.pdb ] || { echo "  receptor PDB missing"; continue; }

  # 3. vdw.defn
  cp -f $D/parameters/vdw_AMBER_parm94.defn ./vdw.defn

  # 4. prepare_amber.pl (-i ignores divcon failure for the ligand; -c uses existing charges)
  $D/bin/prepare_amber.pl -i -c ${lig}.mol2 ${rec}.pdb > prepare_amber.out 2>&1 || true

  # 5. Run amberize_ligand manually (-c gas bypasses divcon)
  $D/bin/amberize_ligand ${lig}.1 -nc 0 -c gas > amberize_ligand.log 2>&1 || true
  if [ ! -s ${lig}.1.prmtop ]; then
    echo "  [FAILED] ${lig}.1.prmtop not generated"
    continue
  fi

  # 6. amberize_complex
  $D/bin/amberize_complex ${rec} ${lig}.1 > amberize_complex.log 2>&1 || true
  if [ ! -s ${rec}.${lig}.1.prmtop ]; then
    echo "  [FAILED] ${rec}.${lig}.1.prmtop not generated"
    continue
  fi

  # 7. Write the official dock.in template
  cat > dock.in <<EOF
conformer_search_type                                        rigid
use_internal_energy                                          no
ligand_atom_file                                             ${lig}.amber_score.mol2
limit_max_ligands                                            no
skip_molecule                                                no
read_mol_solvation                                           no
calculate_rmsd                                               no
use_database_filter                                          no
orient_ligand                                                no
bump_filter                                                  no
score_molecules                                              yes
contact_score_primary                                        no
grid_score_primary                                           no
gist_score_primary                                           no
multigrid_score_primary                                      no
dock3.5_score_primary                                        no
continuous_score_primary                                     no
footprint_similarity_score_primary                           no
pharmacophore_score_primary                                  no
hbond_score_primary                                          no
internal_energy_score_primary                                no
descriptor_score_primary                                     no
gbsa_zou_score_primary                                       no
gbsa_hawkins_score_primary                                   no
SASA_score_primary                                           no
amber_score_primary                                          yes
amber_score_receptor_file_prefix                              ${rec}
amber_score_movable_region                                   ligand
amber_score_minimization_rmsgrad                             0.01
amber_score_before_md_minimization_cycles                    100
amber_score_md_steps                                         3000
amber_score_after_md_minimization_cycles                     100
amber_score_gb_model                                         5
amber_score_nonbonded_cutoff                                 18.0
amber_score_temperature                                      300.0
amber_score_abort_on_unprepped_ligand                        yes
ligand_outfile_prefix                                        output
write_mol_solvation                                          no
write_orientations                                           no
num_final_scored_poses                                       1
num_preclustered_conformers                                  1
score_threshold                                              100.0
rank_ligands                                                 no
EOF

  # 8. Run dock6 Amber Score
  $D/bin/dock6 -i dock.in -o dock.out -v > dock6.stdout 2>&1 || true

  # 9. Extract results
  if grep -q "Amber_Score:" dock.out 2>/dev/null; then
    echo "  ✓ $(grep 'Amber_Score:' dock.out | tail -1 | tr -s ' ')"
  else
    echo "  [FAILED] Amber_Score not found"
    tail -5 dock.out
  fi
done

echo ""
echo "================================================================"
echo "  Summary"
echo "================================================================"
for sys in $SYSTEMS; do
  d=$ASDIR/$sys
  s=$(grep 'Amber_Score:' $d/dock.out 2>/dev/null | tail -1 | awk '{print $2}')
  printf "%-28s %s\n" "$sys" "${s:-N/A}"
done
