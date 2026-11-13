#!/bin/bash
# ============================================================
# Run DOCK6 Amber Score for the eight systems (portable)
# Usage:
#   bash 03_run_amberscore.sh
#   bash 03_run_amberscore.sh state8_CNP0022627_1   # single system
# ============================================================

# ---------- Amber environment ----------
AMBER_ENV=${AMBER_ENV:-/home/liuxiaomeng_cluster/software/amber20/amber20_env.sh}
[ -f "$AMBER_ENV" ] && source "$AMBER_ENV"

# ---------- Automatic directory detection ----------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SYSDIR=$SCRIPT_DIR/systems

if ! command -v dock6 >/dev/null 2>&1; then
    echo "Error: dock6 not found. Load DOCK6 or add dock6 to PATH."
    exit 1
fi
echo "Using dock6: $(command -v dock6)"

ALL="state8_CNP0022627_1 state8_CNP0183427_0 state8_CNP0604857_3 state8_CNP0606043_1 \
     state12_CNP0022627_1 state12_CNP0183427_0 state12_CNP0604857_3 state12_CNP0606043_1"
SYSTEMS=${1:-$ALL}

for sys in $SYSTEMS; do
    dir=$SYSDIR/$sys
    [ -f "$dir/dock.in" ] || { echo "Skipping $sys (no dock.in)"; continue; }
    echo ""
    echo "========== Amber Score: $sys =========="
    cd "$dir" || exit 1
    dock6 -i dock.in -o dock.out -v
    if [ -f dock.out ]; then
        echo "  Done"
        grep -iE "amber score|binding|total" dock.out | tail -5
    else
        echo "  [FAILED] dock.out not found"
    fi
done
echo ""
echo "========== All complete =========="
