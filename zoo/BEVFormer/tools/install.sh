PARTITION=shlab_adg_2
JOB_NAME=install

srun --partition=${PARTITION} \
     --mpi=pmi2 \
     --gres=gpu:1 \
     -n1 \
     --ntasks-per-node=1 \
     --job-name=${JOB_NAME} \
     --cpus-per-task=16 \
     -x SH-IDC1-10-140-1-100 \
     --kill-on-bad-exit=1 \
     pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html --no-cache-dir