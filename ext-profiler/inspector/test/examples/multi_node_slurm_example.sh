#!/bin/bash
#SBATCH -p batch --nodes=32
#SBATCH -t 240:00 --exclusive
#SBATCH --job-name=ncclperftests
#SBATCH --account=your_account
#SBATCH --gres=gpu:8

# Multi-Node NCCL Inspector Example with SLURM
# This script demonstrates how to run NCCL Inspector in a multi-node SLURM environment

# ompi
export OMPI_MCA_btl="tcp,self"
export OMPI_MCA_btl_tcp_if_include="enp90s0f0np0"
export PMIX_MCA_gds=^ds12

export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,enp90s0f0np0
# nccl
export NCCL_IGNORE_CPU_AFFINITY=0
export NCCL_IB_HCA=^mlx5_3,mlx5_4
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=20

#NCCL Lib
export NCCL_ROOT=/path/to/your/nccl/
export NCCL_TESTS_PATH=${NCCL_ROOT}/build/test/perf
export LD_LIBRARY_PATH=${NCCL_ROOT}/build/lib:/path/to/openmpi/lib/:${LD_LIBRARY_PATH}
export NCCL_OUT_DIR=nccl_out_${SLURM_JOB_ID}
mkdir -p ${NCCL_OUT_DIR}

#NCCL Inspector
export NCCL_PROFILER_PLUGIN=${NCCL_ROOT}/ext-profiler/inspector/libnccl-profiler-inspector.so
export NCCL_INSPECTOR_ENABLE=1
export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
export NCCL_INSPECTOR_DUMP_DIR=${NCCL_OUT_DIR}/nccl_insp_out_${SLURM_JOB_ID}
mkdir -p ${NCCL_INSPECTOR_DUMP_DIR}

DATE_TIME_STR=$(date "+%Y-%m-%d_%H:%M:%S.%N")
export NUM_NODES=${SLURM_NNODES}

for num_qps in 1
do
    for iter in 0
    do
        out_file=${NCCL_OUT_DIR}/nccl_test_sendrecv_perf_${NUM_NODES}_${iter}_iter_algo_auto_proto_default_numqps_${num_qps}_${SLURM_JOB_ID}
	srun --comment=sysctl-sys.kernel.numa_balancing=0,transparent_hugepage_defrag=never,transparent_hugepage=never --mpi=pmix -t 120 -N ${NUM_NODES} --ntasks-per-node 8 ${NCCL_TESTS_PATH}/sendrecv_perf -b8 -e16G -f2 -c0 -n 5 -w 5 >& ${out_file}

	for split_mask in 0x7 0x3 0x0
	do
	    out_file=${NCCL_OUT_DIR}/nccl_test_all_reduce_perf_${NUM_NODES}_${iter}_iter_algo_auto_proto_default_split_mask_${split_mask}_numqps_${num_qps}_${SLURM_JOB_ID}
	    NCCL_TESTS_SPLIT_MASK=$split_mask NCCL_IB_QPS_PER_CONNECTION=${num_qps} srun --comment=sysctl-sys.kernel.numa_balancing=0,transparent_hugepage_defrag=never,transparent_hugepage=never --mpi=pmix -t 120 -N ${NUM_NODES} --ntasks-per-node 8 ${NCCL_TESTS_PATH}/all_reduce_perf -c0 -f2 -b8 -e16G -n 5 -w 5 >& ${out_file}
	done

        for split_mask in 0x7 0x3 0x0
	do
	    out_file=${NCCL_OUT_DIR}/nccl_test_all_gather_perf_${NUM_NODES}_${iter}_iter_algo_auto_proto_default_split_mask_${split_mask}_numqps_${num_qps}_${SLURM_JOB_ID}
	    NCCL_TESTS_SPLIT_MASK=$split_mask NCCL_IB_QPS_PER_CONNECTION=${num_qps} srun --comment=sysctl-sys.kernel.numa_balancing=0,transparent_hugepage_defrag=never,transparent_hugepage=never --mpi=pmix -t 120 -N ${NUM_NODES} --ntasks-per-node 8 ${NCCL_TESTS_PATH}/all_gather_perf -c0 -f2 -b8 -e16G -n 5 -w 5 >& ${out_file}
	done

        for split_mask in 0x7 0x3 0x0
	do
	    out_file=${NCCL_OUT_DIR}/nccl_test_reduce_scatter_perf_${NUM_NODES}_${iter}_iter_algo_auto_proto_default_split_mask_${split_mask}_numqps_${num_qps}_${SLURM_JOB_ID}
	    NCCL_TESTS_SPLIT_MASK=$split_mask NCCL_IB_QPS_PER_CONNECTION=${num_qps} srun --comment=sysctl-sys.kernel.numa_balancing=0,transparent_hugepage_defrag=never,transparent_hugepage=never --mpi=pmix -t 120 -N ${NUM_NODES} --ntasks-per-node 8 ${NCCL_TESTS_PATH}/reduce_scatter_perf -c0 -f2 -b8 -e16G -n 5 -w 5 >& ${out_file}
        done

        out_file=${NCCL_OUT_DIR}/nccl_test_alltoall_perf_${NUM_NODES}_${iter}_iter_algo_auto_proto_default_numqps_${num_qps}_${SLURM_JOB_ID}
	srun --comment=sysctl-sys.kernel.numa_balancing=0,transparent_hugepage_defrag=never,transparent_hugepage=never --mpi=pmix -t 120 -N ${NUM_NODES} --ntasks-per-node 8 ${NCCL_TESTS_PATH}/alltoall_perf -b8 -e4G -f2 -c0 -n 5 -w 5 >& ${out_file}
    done
done
