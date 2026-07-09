#!/bin/bash

#SBATCH -J LigandMD         # Job name
#SBATCH --partition=gpu4090     # gpua800
#SBATCH --qos=4gpus
#SBATCH -N 1                    # Single node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1            # 1 GPUs
#SBATCH -o %j.out         # Output result
#SBATCH -e %j.err          # Error output


ml load amber
#istart=1
#iend=3
#I=$istart
#while [ $I -le $iend ]
#do
#  suffix=`printf %03d $I`
#  suffix2=`printf %03d $((I-1))`
#  pmemd.cuda -O -i prod.in -o prod$suffix.out -p wbox.prmtop -c prod$suffix2.rst -x prod$suffix.nc -r prod$suffix.rst
#  I=$((I+1))
#done


# gamd equilibration
pmemd.cuda -O -i gamd-restart-2.in -p complex-amber.top -c gamd-500.rst -o md-1us.out -x md-1us.nc -r gamd-1us.rst -gamd gamd-1us.log

