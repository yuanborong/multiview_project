#!/bin/bash
#SBATCH --job-name=data_adpotor                       # Job name
#SBATCH --partition=sixhour                      # Partition Name (Required)
#SBATCH --mail-type=BEGIN,END,FAIL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yuanborongcs@163.com            # Where to send mail
#SBATCH --ntasks=1                               # Run a single task
#SBATCH --cpus-per-task=8                        # Number of CPU cores per task
#SBATCH --mem-per-cpu=15gb                       # Job memory request
#SBATCH --time=0-06:00:00                        # Time limit days-hrs:min:sec
#SBATCH --output=data_adpotor.log              # Standard output and error log

echo "Running on $SLURM_CPUS_PER_TASK cores"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.7
python /panfs/pfs.local/work/liu/xzhang_sta/yuanborong/data/YFDataAdoptor.py 2010 ;
python /panfs/pfs.local/work/liu/xzhang_sta/yuanborong/data/YFDataAdoptor.py 2011 ;
python /panfs/pfs.local/work/liu/xzhang_sta/yuanborong/data/YFDataAdoptor.py 2012 ;
python /panfs/pfs.local/work/liu/xzhang_sta/yuanborong/data/YFDataAdoptor.py 2013 ;
