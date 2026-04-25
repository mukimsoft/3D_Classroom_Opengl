cmake_minimum_required(VERSION 3.16)
project(Classroom3D CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If GLAD is already installed in your lab setup, this is enough.
# Otherwise, add glad.c to the project and include the glad include folder.

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(Classroom3D
    main.cpp
    # glad.c
)

target_link_libraries(Classroom3D PRIVATE glfw OpenGL::GL)
target_include_directories(Classroom3D PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
