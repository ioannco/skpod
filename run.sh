#BSUB -W 00:29
#BSUB -o "res.txt"
#BSUB -e "my_job.%J.err"
#BSUB -R "affinity[core(1)]"
source /polusfs/setenv/setup.SMPI
for opt in 0 1 2 3 fast
do
        for y in MINI_DATASET SMALL_DATASET MEDIUM_DATASET LARGE_DATASET
        do
                for x in 1 2 3 4 5 6 7 8 9 10 20 40 60 80 100 120 140 160
                do
                        echo -n "mpi $y $opt $x "
                        mpicc -D$y MPI.c -o mpi
                        mpirun -np $x --oversubscribe ./mpi --over
                done
        done
done