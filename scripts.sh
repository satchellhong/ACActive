
#!/bin/bash


for start in 13 14
do
  CUDA_VISIBLE_DEVICES=5 taskset -c 45-49 python experiments/main.py -dataset ALDH1 -acq exploitation -arch mlp -start $start
done
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,7 taskset -c 10-14 python experiments/main.py -dataset ALDH1 -acq bald -arch mlp