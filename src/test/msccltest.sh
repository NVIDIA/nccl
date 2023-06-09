#!/bin/sh

echo ""
echo "Starting the msccl tests, which will including below algo"
echo "allreduce_a100_allpairs, allreduce_a100_ring, alltoall_allpairs"
echo "with both nvlink and ib (NCCL_P2P_DISABLE=1, NCCL_SHM_DISABLE=1) and run both graph (-G 1) and non-graph mode in nccl-tests"
echo ""

declare MSCCL_PROTO
declare MSCCL_DATA_TYPE 
declare MSCCL_OP_TYPE
declare MSCCL_XML_FILES_PARAM
declare MSCCL_ALGOS
declare MSCCL_PATH
declare NCCL_LIB
declare NCCL_ALGO
declare NCCL_TESTS_PATH
declare TEST_RESULT_SUB_PATH
declare MSCCL_ALGO_TEST_PATH
declare ITERATION_COUNT
declare TOPO_FILE_PARAM
declare GRAPH_FILE_PARAM
declare CUDA_VERSION
declare CUDA_ARCH_CODE

#test cases
declare NCCL_P2P
declare NCCL_SHM
declare CUDA_GRAPH
declare ONE_PROCESS
declare WARM_UP_COUNT

MSCCL_TOOL=$HOME/msccl-tools/examples/mscclang
MSCCL_ALGO_PATH=$HOME/msccl-algo
TESTRESULT_HOME=$HOME/msccl-test-results
NCCL_TEST_ORIGIN=$HOME/nccl-tests-original
MSCCL_HOME=$HOME/nccl-218/nccl

TEST_TYPE=${1:-perf}
NUM_GPUS=${2:-8}

if ! [[ $NUM_GPUS =~ ^[0-9]+$ && $NUM_GPUS -ge 1 && $NUM_GPUS -le 16 ]]; then
    echo "invalid input of NUM_GPUS: $NUM_GPUS, should be a number between 1 and 16"
    exit 1
elif [ $NUM_GPUS -eq 4 ]; then
    TOPO_FILE_PARAM="-x NCCL_TOPO_FILE=$HOME/nccl/src/test/ncv4/topo.xml"
    GRAPH_FILE_PARAM="-x NCCL_GRAPH_FILE=$HOME/nccl/src/test/ncv4/graph.xml"
fi

if [ $TEST_TYPE = "all" ]; then
    MSCCL_ALGOS=(allreduce_a100_allpairs allreduce_a100_ring alltoall_allpairs)
    MSCCL_PROTO=(LL LL128 Simple) 
    MSCCL_DATA_TYPE=(int8 uint8 int32 uint32 int64 uint64 half float double bfloat16 fp8_e4m3 fp8_e5m2)
    MSCCL_OP_TYPE=(sum prod max min avg mulsum)
    NCCL_LIB=(NCCL-218-WITH-MSCCL)
    NCCL_P2P=(0 1)
    NCCL_SHM=(0 1)
    CUDA_GRAPH=(0 1)
    ONE_PROCESS=(0 1)
    WARM_UP_COUNT=0
    ITERATION_COUNT=100
    TEST_RESULT_SUB_PATH=$(date +"%m.%d")-all
elif [ $TEST_TYPE = "perf" ]; then
    MSCCL_ALGOS=(allreduce_a100_allpairs allreduce_a100_ring alltoall_allpairs)
    MSCCL_PROTO=(LL LL128 Simple) 
    MSCCL_DATA_TYPE=(float half bfloat16 fp8_e4m3 fp8_e5m2)
    MSCCL_OP_TYPE=(sum)
    # NCCL_LIB=(MSCCL-212 NCCL-218 NCCL-218-WITH-MSCCL)
    NCCL_LIB=(NCCL-218-WITH-MSCCL)
    NCCL_P2P=(1)
    NCCL_SHM=(1)
    CUDA_GRAPH=(0 1)
    ONE_PROCESS=(0)
    WARM_UP_COUNT=0
    ITERATION_COUNT=100
    TEST_RESULT_SUB_PATH=$(date +"%m.%d")-perf
    if [ ! -d "$NCCL_TEST_ORIGIN" ]; then
        mkdir $NCCL_TEST_ORIGIN
        cd $NCCL_TEST_ORIGIN
        git clone https://github.com/NVIDIA/nccl-tests.git
        cd $HOME
    fi
else
    echo "The test type: $TEST_TYPE is not supported"
    exit 1
fi

CUDA_VERSION=$(nvidia-smi | grep "CUDA Version:" | awk '{print $9}')
if [ $(echo "$CUDA_VERSION >= 12.1" | bc) -eq 1 ]; then
    CUDA_ARCH_CODE=90
else
    CUDA_ARCH_CODE=80
fi

if [ ! -d "$MSCCL_ALGO_PATH" ]; then
    mkdir $MSCCL_ALGO_PATH
fi

if [ ! -d "$MSCCL_TOOL" ]; then
    git clone https://github.com/microsoft/msccl-tools.git
    cd msccl-tools
    pip install .
    cd ..
fi

TESTRESULT_HOME=$TESTRESULT_HOME/$TEST_RESULT_SUB_PATH

if [ ! -d "$TESTRESULT_HOME" ]; then
    mkdir -p $TESTRESULT_HOME
fi

for lib in ${NCCL_LIB[@]}; do
    if [ $lib = "MSCCL-212" ]; then
        if [ ! -d "$HOME/msccl" ]; then
            git clone https://github.com/microsoft/msccl.git
        fi
        MSCCL_PATH=$HOME/msccl/build
        if [ ! -d "MSCCL_PATH" ]; then
            cd msccl
            make -j src.build NVCC_GENCODE="-gencode=arch=compute_$CUDA_ARCH_CODE,code=sm_$CUDA_ARCH_CODE"
            cd ..
        fi
        MSCCL_ALGO_TEST_PATH=$MSCCL_PATH/lib/msccl-algorithms
        MSCCL_XML_FILES_PARAM="-x MSCCL_XML_FILES=$MSCCL_ALGO_TEST_PATH/test.xml"
        NCCL_ALGO=MSCCL,RING,TREE
        NCCL_TESTS_PATH=$NCCL_TEST_ORIGIN/nccl-tests
    elif [ $lib = "NCCL-218" ]; then
        if [ ! -d "$HOME/nccl-218" ]; then
            cd $HOME
            git clone https://github.com/NVIDIA/nccl.git
            cd ..
        fi
        MSCCL_PATH=$HOME/nccl-218/build
        if [ ! -d "$MSCCL_PATH" ]; then
            cd $HOME/nccl-218
            make -j src.build NVCC_GENCODE="-gencode=arch=compute_$CUDA_ARCH_CODE,code=sm_$CUDA_ARCH_CODE"
            cd ..
        fi
        MSCCL_ALGO_TEST_PATH=$MSCCL_PATH/lib/msccl-algorithms
        MSCCL_XML_FILES_PARAM=""
        NCCL_ALGO=RING,TREE
        NCCL_TESTS_PATH=$NCCL_TEST_ORIGIN/nccl-tests
    else
        MSCCL_PATH=$MSCCL_HOME/build
        if [ ! -d "$MSCCL_PATH" ]; then
            cd $MSCCL_HOME
            make -j src.build NVCC_GENCODE="-gencode=arch=compute_$CUDA_ARCH_CODE,code=sm_$CUDA_ARCH_CODE"
            cd ..
        fi
        MSCCL_ALGO_TEST_PATH=$MSCCL_PATH/lib/msccl-algorithms
        MSCCL_XML_FILES_PARAM=""
        NCCL_ALGO=MSCCL,RING,TREE
        NCCL_TESTS_PATH=$HOME/nccl-tests
        if [ ! -d "$NCCL_TESTS_PATH" ]; then
            cd $HOME
            git clone --branch users/liand/enable-msccl-on-nccl https://github.com/C-AI-Inference-Platform/nccl-tests.git
        fi
    fi
    echo "Compiling the nccl test tool with $lib"
    cd $NCCL_TESTS_PATH
    nccl_test_build="make MPI=1 MPI_HOME=/usr/local/mpi NCCL_HOME=$MSCCL_PATH -j"
    echo $nccl_test_build
    eval $nccl_test_build
    wait
    cd ..

    if [ ! -d "$MSCCL_ALGO_TEST_PATH" ]; then
        mkdir $MSCCL_ALGO_TEST_PATH
    fi

    for algo in ${MSCCL_ALGOS[@]}; do
        for proto in ${MSCCL_PROTO[@]}; do
            algo_file=${algo}_${NUM_GPUS}n_$proto.xml
            if [ ! -e $MSCCL_ALGO_PATH/$algo_file ]; then
                echo "The algo file: $algo_file does not exist, generating..."
                msccl_algo_gen="python $MSCCL_TOOL/$algo.py --protocol=$proto $NUM_GPUS 2 > $MSCCL_ALGO_PATH/$algo_file"
                if [ $algo = "allreduce_a100_ring" ]; then
                    msccl_algo_gen="python $MSCCL_TOOL/$algo.py --protocol=$proto $NUM_GPUS 1 2 > $MSCCL_ALGO_PATH/$algo_file"
                fi
                echo $msccl_algo_gen
                eval $msccl_algo_gen
                wait
                sed -i '1s/>/ outofplace="0" minBytes="0" maxBytes="327680">/' $MSCCL_ALGO_PATH/$algo_file
            fi
            echo "Copying the algo file $MSCCL_ALGO_PATH/$algo_file to $MSCCL_ALGO_TEST_PATH/test.xml"
            cp $MSCCL_ALGO_PATH/$algo_file $MSCCL_ALGO_TEST_PATH/test.xml
            wait
            echo "Running the $algo with $proto"
            if [ $algo = "allreduce_a100_allpairs" ] || [ $algo = "allreduce_a100_ring" ]; then
                NCCL_TEST_TYPE=all_reduce_perf
            else
                NCCL_TEST_TYPE=alltoall_perf
            fi
        
            #for perf, only care about the nccl_p2p_disable and nccl_shm_disable
            for NCCL_P2P_DISABLE in ${NCCL_P2P[@]}; do #control the whehter nv link is disabled or not
                for NCCL_SHM_DISABLE in ${NCCL_SHM[@]}; do 
                    for ENABLE_CUDA_GRAPH in ${CUDA_GRAPH[@]}; do
                        for ENABLE_ONE_PROCESS in ${ONE_PROCESS[@]}; do
                            for DATA_TYPE in ${MSCCL_DATA_TYPE[@]}; do
                                for OP_TYPE in ${MSCCL_OP_TYPE[@]}; do
                                    testresult=$TESTRESULT_HOME/${algo}_${proto}_${NCCL_P2P_DISABLE}_${NCCL_SHM_DISABLE}_${ENABLE_CUDA_GRAPH}_${ENABLE_ONE_PROCESS}_${DATA_TYPE}_${OP_TYPE}_${lib}.txt
                                    echo "Running the $algo with $proto with configs: NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE, NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE, ENABLE_CUDA_GRAPH=$ENABLE_CUDA_GRAPH, ENABLE_ONE_PROCESS=$ENABLE_ONE_PROCESS, DATA_TYPE=$DATA_TYPE, OP_TYPE=$OP_TYPE"
                                    if [ $ENABLE_ONE_PROCESS -eq 1 ]; then
                                       msccl_test="mpirun --allow-run-as-root -np 1 -x LD_LIBRARY_PATH=$MSCCL_PATH/lib/:$LD_LIBRARY_PATH -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=INIT,ENV -x NCCL_ALGO=$NCCL_ALGO $MSCCL_XML_FILES_PARAM -x NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE -x NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE $TOPO_FILE_PARAM $GRAPH_FILE_PARAM $NCCL_TESTS_PATH/build/$NCCL_TEST_TYPE -b 5K -e 320K -d $DATA_TYPE -f 2 -g $NUM_GPUS -c 1 -o $OP_TYPE -n $ITERATION_COUNT -w $WARM_UP_COUNT -G $ENABLE_CUDA_GRAPH -z 0"
                                    elif [ $NUM_GPUS -eq 16 ]; then
                                       msccl_test="mpirun --allow-run-as-root --tag-output -map-by ppr:8:node -hostfile /root/nccl-218/src/test/ndv5/hostfile -x LD_LIBRARY_PATH -mca coll_hcoll_enable 0 --bind-to none -x NCCL_TOPO_FILE=/root/nccl-218/src/test/ndv5/ndv5-topo-new.xml -x NCCL_DEBUG=WARN -x LD_PRELOAD=$MSCCL_PATH/lib/libnccl.so:$LD_PRELOAD -x NCCL_MIN_NCHANNELS=32 -x NCCL_IB_QPS_PER_CONNECTION=2 -x NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE -x NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE $NCCL_TESTS_PATH/build/$NCCL_TEST_TYPE -b 5K -e 320K -d $DATA_TYPE -f 2 -g 1 -c 1 -w $WARM_UP_COUNT -n $ITERATION_COUNT -G $ENABLE_CUDA_GRAPH -z 0"  
                                    else
                                       msccl_test="mpirun --allow-run-as-root -np $NUM_GPUS -x LD_LIBRARY_PATH=$MSCCL_PATH/lib/:$LD_LIBRARY_PATH -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=INIT,ENV -x NCCL_ALGO=$NCCL_ALGO $MSCCL_XML_FILES_PARAM -x NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE -x NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE $TOPO_FILE_PARAM $GRAPH_FILE_PARAM $NCCL_TESTS_PATH/build/$NCCL_TEST_TYPE -b 5K -e 320K -d $DATA_TYPE -f 2 -g 1 -c 1 -o $OP_TYPE -n $ITERATION_COUNT -w $WARM_UP_COUNT -G $ENABLE_CUDA_GRAPH -z 0"
                                    fi
                                    if [ ! -e $testresult ]; then
                                        echo $msccl_test | tee $testresult
                                        eval $msccl_test >> $testresult
                                        wait
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done