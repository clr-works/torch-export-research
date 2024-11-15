cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(aoti_example)

# Find Torch
if(NOT DEFINED TORCH_DIR)
  find_package(Torch REQUIRED)
else()
  find_package(Torch REQUIRED PATHS ${TORCH_DIR})
endif()


add_executable(aoti_example inference.cpp model.so)

add_custom_command(
    OUTPUT model.so
    COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/model.py
    DEPENDS model.py
)

target_link_libraries(aoti_example "${TORCH_LIBRARIES}")
set_property(TARGET aoti_example PROPERTY CXX_STANDARD 17)