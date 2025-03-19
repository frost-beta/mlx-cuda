enable_language(CUDA)

set_target_properties(
  mlx
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON
             # Compute capability 6 is required for synchronization between
             # CPU/GPU with managed memory.
             # TODO: Add more architectures for release build.
             CUDA_ARCHITECTURES "native;60")

find_package(CUDAToolkit REQUIRED)
target_include_directories(mlx PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(mlx PRIVATE CUDA::cudart_static)

target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/event.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/metal.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp)
