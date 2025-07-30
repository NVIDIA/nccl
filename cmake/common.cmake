function(nccl_add_target_options target)
  target_compile_options(${target} PRIVATE $<$<CONFIG:Debug>:-ggdb3>)
  target_compile_options(${target} PRIVATE $<$<NOT:$<CONFIG:Debug>>:-O3>)
  target_compile_options(
    ${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -Xptxas
                      -maxrregcount=96 -Xfatbin -compress-all -fPIC>)
  target_compile_options(${target} PRIVATE -fPIC -Wall -Wno-unused-function
                                           -Wno-sign-compare -Wvla)
  set_property(TARGET ${target} PROPERTY CXX_STANDARD 17)
  set_property(TARGET ${target} PROPERTY CUDA_STANDARD 17)
  set_property(TARGET ${target} PROPERTY CXX_VISIBILITY_PRESET hidden)
  set_property(TARGET ${target} PROPERTY VISIBILITY_INLINES_HIDDEN 1)
  set_property(TARGET ${target} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  if(VERBOSE)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas
                                             -v -Xcompiler -Wall,-Wextra>)
    target_compile_options(${target} PRIVATE -Wall -Wextra)
  endif()

  if(TRACE)
    target_compile_options(${target} PRIVATE ENABLE_TRACE)
  endif()

  if(NOT NVTX)
    target_compile_options(${target} PRIVATE NVTX_DISABLE)
  endif()

  if(KEEP)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-keep>)
  endif()

  if(PROFAPI)
    target_compile_options(${target} PRIVATE PROFAPI)
  endif()

  if(NET_PROFILER)
    target_compile_options(${target} PRIVATE NET_PROFILER)
  endif()
endfunction()
