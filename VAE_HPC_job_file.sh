#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J VAE_test
### -- ask for number of cores (default: 1) -- 
#BSUB -n 12
###BSUB -gpu "num=1:mode=exclusive_process" 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=1GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 3GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 72:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s194698@student.dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output/Output_%J.out 
#BSUB -e Error/Error_%J.err 

cd $PBS_O_WORKDIR
###module load cudnn/v7.6.5.32-prod-cuda-10.1

cd /work3/s194698/deep_learning/ && CPU_VAE.sh > output.out
