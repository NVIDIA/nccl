#!/bin/bash
#SBATCH -p batch --account=your_account -t 00:45:00 --nodes=32 --exclusive --mem=0 --ntasks-per-node=8 --gpus-per-node=8 --dependency=singleton --job-name=training_with_inspector

# Distributed Training with NCCL Inspector Example
# This script demonstrates how to run NCCL Inspector with a real distributed training workload

# Standard NCCL and training environment variables
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export UB_TIMEOUT=720
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_FUSED_ATTN=0  # Disable cuDNN fused attention.
export NCCL_DEBUG=INFO

# NCCL Inspector Configuration
export NCCL_PROFILER_PLUGIN=/path/to/nccl/ext-profiler/inspector/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
export NCCL_INSPECTOR_DUMP_DIR=/path/to/logs/${SLURM_JOB_ID}/
mkdir -p ${NCCL_INSPECTOR_DUMP_DIR}

# Container and training setup
IMAGE_PATH="/path/to/your/training/image.sqsh"

DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
NAME="training_example"

# Setup directories
RUN_DIR="/path/to/your/training/run/directory"
mkdir -p ${RUN_DIR}
LOGS_DIR="${RUN_DIR}/logs/${NAME}_${SLURM_NNODES}/"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
DATACACHE_DIR="${RUN_DIR}/data_cache"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"

# Mamba triton cache
export TRITON_CACHE_DIR="${RUN_DIR}/triton_cache"
export TRITON_CACHE_MANAGER="megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager"

mkdir -p ${LOGS_DIR} ${CHECKPOINT_DIR} ${DATACACHE_DIR} ${TENSORBOARD_DIR}

# Training configuration
TOKENIZER_MODEL="/path/to/your/tokenizer/model.json"
BLEND_PATH="/path/to/your/blend/files.json"

# Training options (example for a transformer model)
options=" \
    --distributed-timeout-minutes 60 \
    --use-mcore-models \
    --data-cache-path ${DATACACHE_DIR} \
    --no-mmap-bin-files \
    --sequence-parallel \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --init-method-std 0.014 \
    --position-embedding-type none \
    --squared-relu \
    --num-layers 52 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --ffn-hidden-size 21504 \
    --kv-channels 128 \
    --normalization RMSNorm \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --micro-batch-size 6 \
    --global-batch-size 768 \
    --train-samples 1831054688 \
    --lr-decay-samples 1830030688 \
    --lr-warmup-samples 1024000 \
    --lr 8e-4 \
    --min-lr 8e-6 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --lr-decay-style cosine \
    --log-interval 100 \
    --eval-iters 14 \
    --eval-interval 2000 \
    --tokenizer-type TikTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --save-interval 1000 \
    --save-retain-interval 10000 \
    --ckpt-format torch_dist \
    --ckpt-fully-parallel-save \
    --ckpt-fully-parallel-load \
    --async-save \
    --ckpt-assume-constant-structure \
    --log-progress  \
    --timing-log-option minmax \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --log-throughput \
    --bf16 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --tp-comm-overlap \
    --no-create-attention-mask-in-dataloader \
    --manual-gc \
    --num-workers 8 \
    --log-straggler \
    --disable-straggler-on-startup \
    --straggler-minmax-count 16 \
    --check-weight-hash-across-dp-replicas-interval 20000 \
    --tensorboard-dir ${TENSORBOARD_DIR}"

# Launch training with NCCL Inspector enabled
run_cmd="python -u /path/to/your/training/script.py ${options}"

srun -l \
    --container-image "${IMAGE_PATH}" \
    --container-mounts "/home:/home,/path/to/shared:/path/to/shared" \
    --output="${LOGS_DIR}/%x_%j_${DATETIME}.log" \
    sh -c "${run_cmd}"
