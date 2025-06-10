[115/130] /opt/llvm_toolchain/bin/clang++
-I/mnt/data/__wip/nccl-cmake/src/device/../include
-I/mnt/data/__wip/nccl-cmake/src/device
-I/mnt/data/__wip/nccl-cmake/build/src/device
-I/mnt/data/__wip/nccl-cmake/build/src/include
-isystem /usr/local/cuda/include
-O3
-mcpu=neoverse-n1 -mno-outline-atomics
-flto=thin -ffat-lto-objects -Wl,--lto-whole-program-visibility
-mllvm -polly -mllvm -polly-vectorizer=stripmine
-pipe -Qunused-arguments -fident -fcolor-diagnostics
-Wnvcc-compat
-Xcuda-ptxas -maxrregcount=96
-Xcuda-fatbinary -compress-all
-fvisibility=hidden
-fPIC
-O3
-DNDEBUG
-std=gnu++23
--cuda-gpu-arch=sm_52
--cuda-path=/usr/local/cuda
-fPIC
-fcolor-diagnostics
-MD
-MT src/device/CMakeFiles/colldevice.dir/reduce_scatter_sum_u8.cu.o
-MF src/device/CMakeFiles/colldevice.dir/reduce_scatter_sum_u8.cu.o.d
-x cuda
-fgpu-rdc
-c /mnt/data/__wip/nccl-cmake/build/src/device/reduce_scatter_sum_u8.cu
-o src/device/CMakeFiles/colldevice.dir/reduce_scatter_sum_u8.cu.o

export CUDAFLAGS="
-### -O3 \
-flto=thin -ffat-lto-objects -Wl,--lto-whole-program-visibility \
-mcpu=neoverse-n1 \
-Xclang=-mno-outline-atomics \
-mllvm=-polly -mllvm=-polly-vectorizer=stripmine \
-pipe -Qunused-arguments -fident -fcolor-diagnostics \
-Wno-cuda-compat
"

export CUDAFLAGS="
-O3 \
-flto=thin -ffat-lto-objects -Wl,--lto-whole-program-visibility \
-mcpu=neoverse-n1 -mno-outline-atomics \
-mllvm=-polly -mllvm=-polly-vectorizer=stripmine \
-pipe -Qunused-arguments -fident -fcolor-diagnostics \
-Wno-cuda-compat
"

export EUGO_CMAKE_COMMON_OPTIONS="\
-DCMAKE_MESSAGE_LOG_LEVEL=debug \
-DCMAKE_COLOR_DIAGNOSTICS=ON \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_C_STANDARD='17' \
-DCMAKE_C_EXTENSIONS=ON \
-DCMAKE_CXX_STANDARD='23' \
-DCMAKE_CXX_EXTENSIONS=ON \
-DCMAKE_CUDA_STANDARD='23' \
-DCMAKE_CUDA_EXTENSIONS=ON \
-DCMAKE_CUDA_RUNTIME_LIBRARY='Shared' \
-DCMAKE_PREFIX_PATH='/usr/local/lib/python3.12/site-packages;/opt/llvm_toolchain;/usr/local' \
-DCMAKE_MODULE_PATH='/usr/local/share/glog/cmake;/usr/local/share/WebP/cmake;/usr/local/lib64/aws-c_common/cmake;/usr/local/lib64/aws-c_cal/cmake;/usr/local/lib64/s2n/cmake;/usr/local/lib64/aws-c_io/cmake;/usr/local/lib64/aws-c_compression/cmake;/usr/local/lib64/aws-c_http/cmake;/usr/local/lib64/aws-c_sdkutils/cmake;/usr/local/lib64/aws-c_auth/cmake;/usr/local/lib64/aws-checksums/cmake;/usr/local/lib64/aws-c_event_stream/cmake;/usr/local/lib64/aws-c_mqtt/cmake;/usr/local/lib64/aws-c_s3/cmake;/usr/local/lib64/aws-c_iot/cmake;/usr/local/lib64/awc-crt-cpp/cmake;/usr/local/share/graphite2/;/usr/local/share/cpuinfo/;/usr/local/lib64/uuid;/usr/local/share/glog/cmake;/usr/local/share/WebP/cmake;/usr/local/lib64/aws-c_common/cmake;/usr/local/lib64/aws-c_cal/cmake;/usr/local/lib64/s2n/cmake;/usr/local/lib64/aws-c_io/cmake;/usr/local/lib64/aws-c_compression/cmake;/usr/local/lib64/aws-c_http/cmake;/usr/local/lib64/aws-c_sdkutils/cmake;/usr/local/lib64/aws-c_auth/cmake;/usr/local/lib64/aws-checksums/cmake;/usr/local/lib64/aws-c_event_stream/cmake;/usr/local/lib64/aws-c_mqtt/cmake;/usr/local/lib64/aws-c_s3/cmake;/usr/local/lib64/aws-c_iot/cmake;/usr/local/lib64/awc-crt-cpp/cmake;/usr/local/share/graphite2/;/usr/local/share/cpuinfo/;/usr/local/lib64/uuid;/usr/local/lib/python3.12/site-packages/pybind11/share/cmake;/opt/llvm_toolchain/lib/aarch64-unknown-linux-gnu/cmake;/opt/llvm_toolchain/lib/cmake;/usr/local/cuda/lib64/cmake;/usr/local/lib64/cmake;/usr/local/lib/cmake;/usr/local/share/cmake;/usr/local/cmake' \
-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
-DCMAKE_GENERATOR=Ninja"



======
bash-5.2# llvm-cxxfilt _Z31ncclDevKernel_AllGather_RING_LL24ncclDevKernelArgsStorageILm4096EE
ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)
======


-- Check for working CUDA compiler: /opt/llvm_toolchain/bin/clang++ - broken
CMake Error at /usr/local/lib/python3.12/site-packages/cmake/data/share/cmake-3.31/Modules/CMakeTestCUDACompiler.cmake:59 (message):
  The CUDA compiler

    "/opt/llvm_toolchain/bin/clang++"

  is not able to compile a simple test program.

  It fails with the following output:

    Change Dir: '/mnt/data/__wip/nccl-cmake/build/CMakeFiles/CMakeScratch/TryCompile-iq49Yb'

    Run Build Command(s): /usr/local/bin/ninja -v cmTC_393a6
    [1/2] /opt/llvm_toolchain/bin/clang++    -### -O3 -flto=thin -ffat-lto-objects -Wl,--lto-whole-program-visibility -mcpu=neoverse-n1 -Xclang=-mno-outline-atomics -mllvm=-polly -mllvm=-polly-vectorizer=stripmine -pipe -Qunused-arguments -fident -fcolor-diagnostics -Wno-cuda-compat  -std=gnu++23 --cuda-gpu-arch=sm_52 --cuda-path=/usr/local/cuda -fPIE -MD -MT CMakeFiles/cmTC_393a6.dir/main.cu.o -MF CMakeFiles/cmTC_393a6.dir/main.cu.o.d -x cuda -c /mnt/data/__wip/nccl-cmake/build/CMakeFiles/CMakeScratch/TryCompile-iq49Yb/main.cu -o CMakeFiles/cmTC_393a6.dir/main.cu.o
    clang version 20.0.0git
    Target: aarch64-unknown-linux-gnu
    Thread model: posix
    InstalledDir: /opt/llvm_toolchain/bin
    clang++: warning: CUDA version 12.6 is only partially supported [-Wunknown-cuda-version]

    # Device code compilation:
    "/opt/llvm_toolchain/bin/clang-20"
    "-cc1"
    "-triple" "nvptx64-nvidia-cuda"
    "-aux-triple" "aarch64-unknown-linux-gnu"
    "-S" "-disable-free"
    "-clear-ast-before-backend"
    "-disable-llvm-verifier"
    "-discard-value-names"
    "-main-file-name" "main.cu"
    "-mrelocation-model" "pic"
    "-pic-level" "2"
    "-pic-is-pie"
    "-mframe-pointer=all"
    "-fno-rounding-math"
    "-no-integrated-as"
    "-aux-target-cpu" "neoverse-n1"
    "-aux-target-feature" "+v8.2a"
    "-aux-target-feature" "+aes"
    "-aux-target-feature" "+crc"
    "-aux-target-feature" "+dotprod"
    "-aux-target-feature" "+fp-armv8"
    "-aux-target-feature" "+fullfp16"
    "-aux-target-feature" "+lse"
    "-aux-target-feature" "+neon"
    "-aux-target-feature" "+perfmon"
    "-aux-target-feature" "+ras"
    "-aux-target-feature" "+rcpc"
    "-aux-target-feature" "+rdm"
    "-aux-target-feature" "+sha2"
    "-aux-target-feature" "+spe"
    "-aux-target-feature" "+ssbs"
    "-fcuda-is-device"
    "-mllvm" "-enable-memcpyopt-without-libcalls"
    "-fcuda-allow-variadic-functions"
    "-mlink-builtin-bitcode" "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc"
    "-target-sdk-version=12.6"
    "-target-cpu" "sm_52"
    "-target-feature" "+ptx85"
    "-debugger-tuning=gdb"
    "-fno-dwarf-directory-asm"
    "-fdebug-compilation-dir=/mnt/data/__wip/nccl-cmake/build/CMakeFiles/CMakeScratch/TryCompile-iq49Yb"
    "-resource-dir" "/opt/llvm_toolchain/lib/clang/20"
    "-dependency-file" "CMakeFiles/cmTC_393a6.dir/main.cu.o.d"
    "-MT" "CMakeFiles/cmTC_393a6.dir/main.cu.o"
    "-sys-header-deps"
    "-internal-isystem" "/opt/llvm_toolchain/lib/clang/20/include/cuda_wrappers"
    "-include" "__clang_cuda_runtime_wrapper.h"
    "-internal-isystem" "/opt/llvm_toolchain/bin/../include/aarch64-unknown-linux-gnu/c++/v1"
    "-internal-isystem" "/opt/llvm_toolchain/bin/../include/c++/v1"
    "-internal-isystem" "/opt/llvm_toolchain/bin/../include/aarch64-unknown-linux-gnu/c++/v1"
    "-internal-isystem" "/opt/llvm_toolchain/bin/../include/c++/v1"
    "-internal-isystem" "/opt/llvm_toolchain/lib/clang/20/include"
    "-internal-isystem" "/usr/local/include"
    "-internal-externc-isystem" "/include"
    "-internal-externc-isystem" "/usr/include"
    "-internal-isystem" "/usr/local/cuda/include"
    "-internal-isystem" "/opt/llvm_toolchain/lib/clang/20/include"
    "-internal-isystem" "/usr/local/include"
    "-internal-externc-isystem" "/include"
    "-internal-externc-isystem" "/usr/include"
    "-O3"
    "-Wno-cuda-compat"
    "-std=gnu++23"
    "-fdeprecated-macro"
    "-fno-autolink"
    "-ferror-limit" "19"
    "-fno-signed-char"
    "-fgnuc-version=4.2.1"
    "-fno-implicit-modules"
    "-fskip-odr-check-in-gmf"
    "-fcxx-exceptions"
    "-fexceptions"
    "-fcolor-diagnostics"
    "-vectorize-loops"
    "-vectorize-slp"
    "-mno-outline-atomics"
    "-mllvm" "-polly" "-mllvm" "-polly-vectorizer=stripmine"
    "-cuid=37201e22adb59f5e"
    "-D__GCC_HAVE_DWARF2_CFI_ASM=1"
    "-o" "/tmp/main-sm_52-7e5392.s"
    "-x" "cuda" "/mnt/data/__wip/nccl-cmake/build/CMakeFiles/CMakeScratch/TryCompile-iq49Yb/main.cu"

    # Device code -> assembly
    "/usr/local/cuda/bin/ptxas"
    "-m64"
    "-O3"
    "--gpu-name" "sm_52"
    "--output-file" "/tmp/main-sm_52-ded96a.o"
    "/tmp/main-sm_52-7e5392.s"

    # Device assembly packaging for embedding (but not the embedding yet!)
    "/usr/local/cuda/bin/fatbinary"
    "-64"
    "--create" "/tmp/main-a85477.fatbin"
    "--image=profile=sm_52,file=/tmp/main-sm_52-ded96a.o"
    "--image=profile=compute_52,file=/tmp/main-sm_52-7e5392.s"

    # Host code compilation:
    "/opt/llvm_toolchain/bin/clang-20"
    "-cc1"
    "-triple" "aarch64-unknown-linux-gnu"
    "-target-sdk-version=12.6"
    "-fcuda-allow-variadic-functions"
    "-aux-triple" "nvptx64-nvidia-cuda"
    "-emit-obj"
    "-disable-free"
    "-clear-ast-before-backend"
    "-disable-llvm-verifier"
    "-discard-value-names"
    "-main-file-name" "main.cu"
    "-mrelocation-model" "pic"
    "-pic-level" "2"
    "-pic-is-pie"
    "-mframe-pointer=non-leaf"
    "-fmath-errno"
    "-ffp-contract=on"
    "-fno-rounding-math"
    "-mconstructor-aliases"
    "-funwind-tables=2"
    "-target-cpu" "neoverse-n1"
    "-target-feature" "+v8.2a"
    "-target-feature" "+aes"
    "-target-feature" "+crc"
    "-target-feature" "+dotprod"
    "-target-feature" "+fp-armv8"
    "-target-feature" "+fullfp16"
    "-target-feature" "+lse"
    "-target-feature" "+neon"
    "-target-feature" "+perfmon"
    "-target-feature" "+ras"
    "-target-feature" "+rcpc"
    "-target-feature" "+rdm"
    "-target-feature" "+sha2"
    "-target-feature" "+spe"
    "-target-feature" "+ssbs"
    "-target-abi" "aapcs"
    "-debugger-tuning=gdb" "-fdebug-compilation-dir=/mnt/data/__wip/nccl-cmake/build/CMakeFiles/CMakeScratch/TryCompile-iq49Yb"
    "-fcoverage-compilation-dir=/mnt/data/__wip/nccl-cmake/build/CMakeFiles/CMakeScratch/TryCompile-iq49Yb"
    "-resource-dir" "/opt/llvm_toolchain/lib/clang/20"
    "-dependency-file" "CMakeFiles/cmTC_393a6.dir/main.cu.o.d"
    "-MT" "CMakeFiles/cmTC_393a6.dir/main.cu.o"
    "-sys-header-deps" "-internal-isystem" "/opt/llvm_toolchain/lib/clang/20/include/cuda_wrappers"
    "-include" "__clang_cuda_runtime_wrapper.h"
    "-internal-isystem" "/opt/llvm_toolchain/bin/../include/aarch64-unknown-linux-gnu/c++/v1"
    "-internal-isystem" "/opt/llvm_toolchain/bin/../include/c++/v1"
    "-internal-isystem" "/opt/llvm_toolchain/bin/../include/aarch64-unknown-linux-gnu/c++/v1"
    "-internal-isystem" "/opt/llvm_toolchain/bin/../include/c++/v1"
    "-internal-isystem" "/opt/llvm_toolchain/lib/clang/20/include"
    "-internal-isystem" "/usr/local/include"
    "-internal-externc-isystem" "/include"
    "-internal-externc-isystem" "/usr/include"
    "-internal-isystem" "/opt/llvm_toolchain/lib/clang/20/include" "-internal-isystem" "/usr/local/include" "-internal-externc-isystem" "/include" "-internal-externc-isystem" "/usr/include" "-internal-isystem" "/usr/local/cuda/include" "-O3" "-Wno-cuda-compat" "-std=gnu++23" "-fdeprecated-macro" "-ferror-limit" "19" "-fno-signed-char" "-fgnuc-version=4.2.1" "-fno-implicit-modules" "-fskip-odr-check-in-gmf" "-fcxx-exceptions" "-fexceptions" "-fcolor-diagnostics" "-vectorize-loops" "-vectorize-slp" "-mno-outline-atomics" "-mllvm" "-polly" "-mllvm" "-polly-vectorizer=stripmine" "-fcuda-include-gpubinary" "/tmp/main-a85477.fatbin" "-cuid=37201e22adb59f5e" "-flto=thin" "-flto-unit" "-ffat-lto-objects" "-ffat-lto-objects" "-target-feature" "+outline-atomics" "-faddrsig" "-D__GCC_HAVE_DWARF2_CFI_ASM=1" "-o" "CMakeFiles/cmTC_393a6.dir/main.cu.o" "-x" "cuda" "/mnt/data/__wip/nccl-cmake/build/CMakeFiles/CMakeScratch/TryCompile-iq49Yb/main.cu"



    # @TODO

    [2/2] : && /opt/llvm_toolchain/bin/clang++ -### -O3 -flto=thin -ffat-lto-objects -Wl,--lto-whole-program-visibility -mcpu=neoverse-n1 -Xclang=-mno-outline-atomics -mllvm=-polly -mllvm=-polly-vectorizer=stripmine -pipe -Qunused-arguments -fident -fcolor-diagnostics -Wno-cuda-compat  --cuda-path=/usr/local/cuda -L/usr/local/cuda/lib64/stubs -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -L/usr/local/lib64 -L/usr/local/lib -Wl,--undefined-version CMakeFiles/cmTC_393a6.dir/main.cu.o -o cmTC_393a6  -lcudadevrt  -lcudart -L"/usr/local/cuda/lib64" && :
    FAILED: cmTC_393a6
    : && /opt/llvm_toolchain/bin/clang++ -### -O3 -flto=thin -ffat-lto-objects -Wl,--lto-whole-program-visibility -mcpu=neoverse-n1 -Xclang=-mno-outline-atomics -mllvm=-polly -mllvm=-polly-vectorizer=stripmine -pipe -Qunused-arguments -fident -fcolor-diagnostics -Wno-cuda-compat  --cuda-path=/usr/local/cuda -L/usr/local/cuda/lib64/stubs -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -L/usr/local/lib64 -L/usr/local/lib -Wl,--undefined-version CMakeFiles/cmTC_393a6.dir/main.cu.o -o cmTC_393a6  -lcudadevrt  -lcudart -L"/usr/local/cuda/lib64" && :
    clang version 20.0.0git
    Target: aarch64-unknown-linux-gnu
    Thread model: posix
    InstalledDir: /opt/llvm_toolchain/bin
    clang++: error: no such file or directory: 'CMakeFiles/cmTC_393a6.dir/main.cu.o'
    "/opt/llvm_toolchain/bin/ld.lld" "-EL" "--hash-style=gnu" "--eh-frame-hdr" "-m" "aarch64linux" "-pie" "-dynamic-linker" "/lib/ld-linux-aarch64.so.1" "-o" "cmTC_393a6" "/lib/../lib64/Scrt1.o" "/lib/../lib64/crti.o" "/opt/llvm_toolchain/lib/clang/20/lib/aarch64-unknown-linux-gnu/clang_rt.crtbegin.o" "-L/usr/local/cuda/lib64/stubs" "-L/usr/local/cuda/lib64" "-L/usr/local/cuda/lib" "-L/usr/local/lib64" "-L/usr/local/lib" "-L/usr/local/cuda/lib64" "-L/opt/llvm_toolchain/bin/../lib/aarch64-unknown-linux-gnu" "-L/opt/llvm_toolchain/lib/clang/20/lib/aarch64-unknown-linux-gnu" "-L/lib/../lib64" "-L/usr/lib/../lib64" "-L/lib" "-L/usr/lib" "--fat-lto-objects" "-plugin-opt=mcpu=neoverse-n1" "-plugin-opt=O3" "-plugin-opt=thinlto" "--lto-whole-program-visibility" "--undefined-version" "-lcudadevrt" "-lcudart" "-lc++" "-lm" "/opt/llvm_toolchain/lib/clang/20/lib/aarch64-unknown-linux-gnu/libclang_rt.builtins.a" "--as-needed" "-lunwind" "--no-as-needed" "-lc" "/opt/llvm_toolchain/lib/clang/20/lib/aarch64-unknown-linux-gnu/libclang_rt.builtins.a" "--as-needed" "-lunwind" "--no-as-needed" "/opt/llvm_toolchain/lib/clang/20/lib/aarch64-unknown-linux-gnu/clang_rt.crtend.o" "/lib/../lib64/crtn.o"
    ninja: build stopped: subcommand failed.





  CMake will not be able to correctly generate this project.
Call Stack (most recent call first):
  CMakeLists.txt:5 (project)


-- Configuring incomplete, errors occurred!




=====



clang++: warning: CUDA version 12.6 is only partially supported [-Wunknown-cuda-version]
    clang++: warning: 'nvptx64' does not support '-mno-outline-atomics'; flag ignored [-Woption-ignored]
    "/opt/llvm_toolchain/bin/clang-20" "-cc1" "-triple" "nvptx64-nvidia-cuda" "-aux-triple" "aarch64-unknown-linux-gnu" "-S" "-disable-free" "-clear-ast-before-backend" "-disable-llvm-verifier" "-discard-value-names" "-main-file-name" "main.cu" "-mrelocation-model" "pic" "-pic-level" "2" "-pic-is-pie" "-mframe-pointer=all" "-fno-rounding-math" "-no-integrated-as" "-aux-target-cpu" "neoverse-n1" "-aux-target-feature" "+v8.2a" "-aux-target-feature" "+aes" "-aux-target-feature" "+crc" "-aux-target-feature" "+dotprod" "-aux-target-feature" "+fp-armv8" "-aux-target-feature" "+fullfp16" "-aux-target-feature" "+lse" "-aux-target-feature" "+neon" "-aux-target-feature" "+perfmon" "-aux-target-feature" "+ras" "-aux-target-feature" "+rcpc" "-aux-target-feature" "+rdm" "-aux-target-feature" "+sha2" "-aux-target-feature" "+spe" "-aux-target-feature" "+ssbs" "-fcuda-is-device" "-mllvm" "-enable-memcpyopt-without-libcalls" "-fcuda-allow-variadic-functions" "-mlink-builtin-bitcode" "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc" "-target-sdk-version=12.6" "-target-cpu" "sm_52" "-target-feature" "+ptx85" "-debugger-tuning=gdb" "-fno-dwarf-directory-asm" "-fdebug-compilation-dir=/mnt/data/__wip/nccl-cmake/build/CMakeFiles/CMakeScratch/TryCompile-Xg7Zfj" "-resource-dir" "/opt/llvm_toolchain/lib/clang/20" "-dependency-file" "CMakeFiles/cmTC_3cfe4.dir/main.cu.o.d" "-MT" "CMakeFiles/cmTC_3cfe4.dir/main.cu.o" "-sys-header-deps" "-internal-isystem" "/opt/llvm_toolchain/lib/clang/20/include/cuda_wrappers" "-include" "__clang_cuda_runtime_wrapper.h" "-internal-isystem" "/opt/llvm_toolchain/bin/../include/aarch64-unknown-linux-gnu/c++/v1" "-internal-isystem" "/opt/llvm_toolchain/bin/../include/c++/v1" "-internal-isystem" "/opt/llvm_toolchain/bin/../include/aarch64-unknown-linux-gnu/c++/v1" "-internal-isystem" "/opt/llvm_toolchain/bin/../include/c++/v1" "-internal-isystem" "/opt/llvm_toolchain/lib/clang/20/include" "-internal-isystem" "/usr/local/include" "-internal-externc-isystem" "/include" "-internal-externc-isystem" "/usr/include" "-internal-isystem" "/usr/local/cuda/include" "-internal-isystem" "/opt/llvm_toolchain/lib/clang/20/include" "-internal-isystem" "/usr/local/include" "-internal-externc-isystem" "/include" "-internal-externc-isystem" "/usr/include" "-O3" "-Wno-cuda-compat" "-std=gnu++23" "-fdeprecated-macro" "-fno-autolink" "-ferror-limit" "19" "-fno-signed-char" "-fgnuc-version=4.2.1" "-fno-implicit-modules" "-fskip-odr-check-in-gmf" "-fcxx-exceptions" "-fexceptions" "-fcolor-diagnostics" "-vectorize-loops" "-vectorize-slp" "-mllvm" "-polly" "-mllvm" "-polly-vectorizer=stripmine" "-cuid=16f4939dbee8d551" "-D__GCC_HAVE_DWARF2_CFI_ASM=1" "-o" "/tmp/main-sm_52-3f0d40.s" "-x" "cuda" "/mnt/data/__wip/nccl-cmake/build/CMakeFiles/CMakeScratch/TryCompile-Xg7Zfj/main.cu"
    "/usr/local/cuda/bin/ptxas" "-m64" "-O3" "--gpu-name" "sm_52" "--output-file" "/tmp/main-sm_52-49081e.o" "/tmp/main-sm_52-3f0d40.s"
    "/usr/local/cuda/bin/fatbinary" "-64" "--create" "/tmp/main-ae35de.fatbin" "--image=profile=sm_52,file=/tmp/main-sm_52-49081e.o" "--image=profile=compute_52,file=/tmp/main-sm_52-3f0d40.s"
    "/opt/llvm_toolchain/bin/clang-20" "-cc1" "-triple" "aarch64-unknown-linux-gnu" "-target-sdk-version=12.6" "-fcuda-allow-variadic-functions" "-aux-triple" "nvptx64-nvidia-cuda" "-emit-obj" "-disable-free" "-clear-ast-before-backend" "-disable-llvm-verifier" "-discard-value-names" "-main-file-name" "main.cu" "-mrelocation-model" "pic" "-pic-level" "2" "-pic-is-pie" "-mframe-pointer=non-leaf" "-fmath-errno" "-ffp-contract=on" "-fno-rounding-math" "-mconstructor-aliases" "-funwind-tables=2" "-target-cpu" "neoverse-n1" "-target-feature" "+v8.2a" "-target-feature" "+aes" "-target-feature" "+crc" "-target-feature" "+dotprod" "-target-feature" "+fp-armv8" "-target-feature" "+fullfp16" "-target-feature" "+lse" "-target-feature" "+neon" "-target-feature" "+perfmon" "-target-feature" "+ras" "-target-feature" "+rcpc" "-target-feature" "+rdm" "-target-feature" "+sha2" "-target-feature" "+spe" "-target-feature" "+ssbs" "-target-abi" "aapcs" "-debugger-tuning=gdb" "-fdebug-compilation-dir=/mnt/data/__wip/nccl-cmake/build/CMakeFiles/CMakeScratch/TryCompile-Xg7Zfj" "-fcoverage-compilation-dir=/mnt/data/__wip/nccl-cmake/build/CMakeFiles/CMakeScratch/TryCompile-Xg7Zfj" "-resource-dir" "/opt/llvm_toolchain/lib/clang/20" "-dependency-file" "CMakeFiles/cmTC_3cfe4.dir/main.cu.o.d" "-MT" "CMakeFiles/cmTC_3cfe4.dir/main.cu.o" "-sys-header-deps" "-internal-isystem" "/opt/llvm_toolchain/lib/clang/20/include/cuda_wrappers" "-include" "__clang_cuda_runtime_wrapper.h" "-internal-isystem" "/opt/llvm_toolchain/bin/../include/aarch64-unknown-linux-gnu/c++/v1" "-internal-isystem" "/opt/llvm_toolchain/bin/../include/c++/v1" "-internal-isystem" "/opt/llvm_toolchain/bin/../include/aarch64-unknown-linux-gnu/c++/v1" "-internal-isystem" "/opt/llvm_toolchain/bin/../include/c++/v1" "-internal-isystem" "/opt/llvm_toolchain/lib/clang/20/include" "-internal-isystem" "/usr/local/include" "-internal-externc-isystem" "/include" "-internal-externc-isystem" "/usr/include" "-internal-isystem" "/opt/llvm_toolchain/lib/clang/20/include" "-internal-isystem" "/usr/local/include" "-internal-externc-isystem" "/include" "-internal-externc-isystem" "/usr/include" "-internal-isystem" "/usr/local/cuda/include" "-O3" "-Wno-cuda-compat" "-std=gnu++23" "-fdeprecated-macro" "-ferror-limit" "19" "-fno-signed-char" "-fgnuc-version=4.2.1" "-fno-implicit-modules" "-fskip-odr-check-in-gmf" "-fcxx-exceptions" "-fexceptions" "-fcolor-diagnostics" "-vectorize-loops" "-vectorize-slp" "-mllvm" "-polly" "-mllvm" "-polly-vectorizer=stripmine" "-fcuda-include-gpubinary" "/tmp/main-ae35de.fatbin" "-cuid=16f4939dbee8d551" "-flto=thin" "-flto-unit" "-ffat-lto-objects" "-ffat-lto-objects" "-target-feature" "-outline-atomics" "-faddrsig" "-D__GCC_HAVE_DWARF2_CFI_ASM=1" "-o" "CMakeFiles/cmTC_3cfe4.dir/main.cu.o" "-x" "cuda" "/mnt/data/__wip/nccl-cmake/build/CMakeFiles/CMakeScratch/TryCompile-Xg7Zfj/main.cu"
    [2/2] : && /opt/llvm_toolchain/bin/clang++ -### -O3 -flto=thin -ffat-lto-objects -Wl,--lto-whole-program-visibility -mcpu=neoverse-n1 -mno-outline-atomics -mllvm=-polly -mllvm=-polly-vectorizer=stripmine -pipe -Qunused-arguments -fident -fcolor-diagnostics -Wno-cuda-compat  --cuda-path=/usr/local/cuda -L/usr/local/cuda/lib64/stubs -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -L/usr/local/lib64 -L/usr/local/lib -Wl,--undefined-version CMakeFiles/cmTC_3cfe4.dir/main.cu.o -o cmTC_3cfe4  -lcudadevrt  -lcudart -L"/usr/local/cuda/lib64" && :
    FAILED: cmTC_3cfe4
    : && /opt/llvm_toolchain/bin/clang++ -### -O3 -flto=thin -ffat-lto-objects -Wl,--lto-whole-program-visibility -mcpu=neoverse-n1 -mno-outline-atomics -mllvm=-polly -mllvm=-polly-vectorizer=stripmine -pipe -Qunused-arguments -fident -fcolor-diagnostics -Wno-cuda-compat  --cuda-path=/usr/local/cuda -L/usr/local/cuda/lib64/stubs -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib -L/usr/local/lib64 -L/usr/local/lib -Wl,--undefined-version CMakeFiles/cmTC_3cfe4.dir/main.cu.o -o cmTC_3cfe4  -lcudadevrt  -lcudart -L"/usr/local/cuda/lib64" && :
    clang version 20.0.0git
    Target: aarch64-unknown-linux-gnu
    Thread model: posix
    InstalledDir: /opt/llvm_toolchain/bin
    clang++: error: no such file or directory: 'CMakeFiles/cmTC_3cfe4.dir/main.cu.o'
    "/opt/llvm_toolchain/bin/ld.lld" "-EL" "--hash-style=gnu" "--eh-frame-hdr" "-m" "aarch64linux" "-pie" "-dynamic-linker" "/lib/ld-linux-aarch64.so.1" "-o" "cmTC_3cfe4" "/lib/../lib64/Scrt1.o" "/lib/../lib64/crti.o" "/opt/llvm_toolchain/lib/clang/20/lib/aarch64-unknown-linux-gnu/clang_rt.crtbegin.o" "-L/usr/local/cuda/lib64/stubs" "-L/usr/local/cuda/lib64" "-L/usr/local/cuda/lib" "-L/usr/local/lib64" "-L/usr/local/lib" "-L/usr/local/cuda/lib64" "-L/opt/llvm_toolchain/bin/../lib/aarch64-unknown-linux-gnu" "-L/opt/llvm_toolchain/lib/clang/20/lib/aarch64-unknown-linux-gnu" "-L/lib/../lib64" "-L/usr/lib/../lib64" "-L/lib" "-L/usr/lib" "--fat-lto-objects" "-plugin-opt=mcpu=neoverse-n1" "-plugin-opt=O3" "-plugin-opt=thinlto" "--lto-whole-program-visibility" "--undefined-version" "-lcudadevrt" "-lcudart" "-lc++" "-lm" "/opt/llvm_toolchain/lib/clang/20/lib/aarch64-unknown-linux-gnu/libclang_rt.builtins.a" "--as-needed" "-lunwind" "--no-as-needed" "-lc" "/opt/llvm_toolchain/lib/clang/20/lib/aarch64-unknown-linux-gnu/libclang_rt.builtins.a" "--as-needed" "-lunwind" "--no-as-needed" "/opt/llvm_toolchain/lib/clang/20/lib/aarch64-unknown-linux-gnu/clang_rt.crtend.o" "/lib/../lib64/crtn.o"
    ninja: build stopped: subcommand failed.