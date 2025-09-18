nohup python -u jsindy_benchmark.py --gpu 4 --seed 0 --num-repeats 32 > log0.out 2>&1 &
nohup python -u jsindy_benchmark.py --gpu 5 --seed 1 --num-repeats 32 > log1.out 2>&1 &
nohup python -u jsindy_benchmark.py --gpu 6 --seed 2 --num-repeats 32 > log2.out 2>&1 &
nohup python -u jsindy_benchmark.py --gpu 7 --seed 3 --num-repeats 32 > log3.out 2>&1 &
