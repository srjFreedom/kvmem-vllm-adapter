cmake_minimum_required(VERSION 3.16)
project(vllm_kvmem_adapter CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # the .a is linked into the pybind11 .so
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# The decoupled, engine-agnostic kvmem library, built from the origin/master worktree so the
# user's feb4742 checkout (the running llama.cpp KVMem) is left untouched.
# kvmem-v018rc1 provides the NVMe disk tier (nvme_kv_tier.hpp) the disk layer needs; the older
# adapter-base worktree predates it. The contract headers this adapter uses are unchanged.
if(DEFINED ENV{KVMA_SRC} AND NOT KVMEM_BASE)
  set(KVMEM_BASE "$ENV{KVMA_SRC}" CACHE PATH "kvmem worktree root")
else()
  set(KVMEM_BASE "$ENV{HOME}/src/kvmem-v018rc1" CACHE PATH "kvmem worktree root")
endif()
# v0.19's DEFAULT Query scorer is page-softmax, whose CUDA object lives in the core's optional
# recall target (`kvmem-recall-cuda`, source src/cuda/page_softmax.cu). Without it the core only has
# the host reference `page_softmax_scores`, which costs ~3e9 MACs per selection at our geometry.
# Enabling CUDA here is what makes the upstream default usable at all.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)     # the pair on this box is TU102 (sm_75)
endif()
enable_language(CUDA)
add_subdirectory(${KVMEM_BASE}/kvmem ${CMAKE_BINARY_DIR}/kvmem)
kvmem_enable_cuda_recall()

# The async page pipeline (KVMEM_ASYNC_TRANSFER arm) needs the CUDA runtime for its transfer
# stream. The sync arm and the contract tests never touch the GPU, but the backend references
# the async transfer, so the whole library links cudart.
find_package(CUDAToolkit REQUIRED)
add_library(vllm_kvmem src/vllm_backend.cpp src/async_page_pipeline.cpp
    src/vllm_sparse_selector.cpp)
target_include_directories(vllm_kvmem PUBLIC include)
target_link_libraries(vllm_kvmem PUBLIC kvmem CUDA::cudart)
target_compile_features(vllm_kvmem PUBLIC cxx_std_17)

enable_testing()
add_executable(vllm_contract_test tests/vllm_contract_test.cpp)
target_link_libraries(vllm_contract_test PRIVATE vllm_kvmem)
target_include_directories(vllm_contract_test PRIVATE tests)
add_test(NAME vllm_contract_test COMMAND vllm_contract_test)

# ---------------------------------------------------------------------------
# pybind11 extension, built against the vLLM serving venv's Python 3.12.
# ---------------------------------------------------------------------------
if(DEFINED ENV{VLLM_PY})
  set(_VENV_PY "$ENV{VLLM_PY}")
else()
  set(_VENV_PY "$ENV{HOME}/vLLM-2080Ti-Definitive/.venv/bin/python")
endif()
if(EXISTS "${_VENV_PY}")
  set(Python_EXECUTABLE "${_VENV_PY}" CACHE FILEPATH "venv python" FORCE)
  find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
  # pybind11's CMake config ships inside the venv's site-packages (derive from the venv python).
  if(NOT DEFINED ENV{PYBIND11_DIR})
    get_filename_component(_VENV_BIN "${_VENV_PY}" DIRECTORY)
    get_filename_component(_VENV_ROOT "${_VENV_BIN}" DIRECTORY)
    file(GLOB _PYBIND11_CFG "${_VENV_ROOT}/lib/python3.*/site-packages/pybind11/share/cmake/pybind11")
    list(LENGTH _PYBIND11_CFG _PYBIND11_N)
    if(_PYBIND11_N GREATER 0)
      list(GET _PYBIND11_CFG 0 pybind11_DIR)
    endif()
  else()
    set(pybind11_DIR "$ENV{PYBIND11_DIR}")
  endif()
  set(pybind11_DIR "${pybind11_DIR}" CACHE PATH "" FORCE)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(vllm_kvmem_ext MODULE src/vllm_kvmem_ext.cpp)
  target_link_libraries(vllm_kvmem_ext PRIVATE vllm_kvmem)
  # Land the built .so where the venv can import it.
  set_target_properties(vllm_kvmem_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python")
else()
  message(STATUS "venv python not found; skipping vllm_kvmem_ext")
endif()

add_executable(sparse_selector_test tests/sparse_selector_test.cpp)
target_link_libraries(sparse_selector_test PRIVATE vllm_kvmem)
