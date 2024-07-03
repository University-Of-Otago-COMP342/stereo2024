CMAKE_MINIMUM_REQUIRED(VERSION "3.12")

set(CMAKE_CXX_STANDARD 14)

#set(CMAKE_TOOLCHAIN_FILE "K:\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake")

project(Stereo)
find_package (OpenCV CONFIG REQUIRED)
include_directories (${OpenCV_INCLUDE_DIRS})
add_executable(stereo stereo.cpp)
target_link_libraries(stereo
   ${OpenCV_LIBS}
)
