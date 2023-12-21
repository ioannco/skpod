#BSUB -n 20
#BSUB -W 00:29
#BSUB -o "res2.txt"
#BSUB -e "my_job.%J.err"
#BSUB -R "affinity[core(1)]"
for opt in 0 1 2 3 fast
do
        for y in MINI_DATASET SMALL_DATASET MEDIUM_DATASET LARGE_DATASET
        do
                for x in 1 2 3 4 5 6 7 8 
                do
                        echo -n "for $y $opt $x "
                        gcc  for.c -D$y -DNUM_THREADS=$x -O$opt -fopenmp -o  for
                        ./for
                        echo ""
                done
        done
done