#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
##SBATCH --time=00:00:30
#SBATCH --mem=200MB
#SBATCH --output=CoLinAdapt_0.5_810.output
#SBATCH --error=CoLinAdapt_0.5_810.error

cd /if15/lg5bt/IR_Base_CoLinAdapt_0.5/
./run
