find_program(UV_EXECUTABLE uv DOC "Path to uv executable" REQUIRED)
set(
  UV_ENV
  UV_PYTHON_PREFERENCE=only-managed
  UV_PYTHON=3.13
)

set(NCCL4PY_SUPPORTED_CUDA_MAJOR 12 13)
set(NCCL4PY_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(NCCL4PY_DIST_DIR "${CMAKE_BINARY_DIR}/dist")

if(NOT "${CUDAToolkit_VERSION_MAJOR}" IN_LIST NCCL4PY_SUPPORTED_CUDA_MAJOR)
  message(
    FATAL_ERROR
      "Only CUDA ${NCCL4PY_SUPPORTED_CUDA_MAJOR} are supported. Found CUDA ${CUDAToolkit_VERSION_MAJOR}"
  )
endif()

add_custom_target(
  nccl4py_dev
  COMMAND
    ${CMAKE_COMMAND} -E env ${UV_ENV}
    "${UV_EXECUTABLE}" sync --group test --extra cu${CUDAToolkit_VERSION_MAJOR}
  COMMAND
    ${CMAKE_COMMAND} -E env ${UV_ENV}
    "${UV_EXECUTABLE}" pip install -i "https://download.pytorch.org/whl/cu${CUDAToolkit_VERSION_MAJOR}${CUDAToolkit_VERSION_MINOR}" torch
  COMMAND
    ${CMAKE_COMMAND} -E env ${UV_ENV}
    "${UV_EXECUTABLE}" pip install cupy-cuda${CUDAToolkit_VERSION_MAJOR}x~=13.6
  WORKING_DIRECTORY "${NCCL4PY_DIR}"
  COMMENT "Installing nccl4py with cu${CUDAToolkit_VERSION_MAJOR} extra and test dependencies"
)

add_custom_target(
  nccl4py
  COMMAND
    ${CMAKE_COMMAND} -E env ${UV_ENV}
    "${UV_EXECUTABLE}" run python --version
  COMMAND
    ${CMAKE_COMMAND} -E env ${UV_ENV}
    "${UV_EXECUTABLE}" build --out-dir "${NCCL4PY_DIST_DIR}" "${NCCL4PY_DIR}"
  WORKING_DIRECTORY "${NCCL4PY_DIR}"
  COMMENT "Building nccl4py wheel into ${NCCL4PY_DIST_DIR}"
)
