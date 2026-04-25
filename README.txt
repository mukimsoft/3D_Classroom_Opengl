3D Classroom (OpenGL)

Files:
- main.cpp
- CMakeLists.txt

Build idea:
1. Put glad.c in the same folder if your setup needs it.
2. Make sure glad/include and glfw are available.
3. Run:
   cmake -S . -B build
   cmake --build build

Controls:
- ESC to close

Scene includes:
- room walls, floor, ceiling
- student desks and chairs
- teacher table
- whiteboard
- projector
- AC unit

Textures are generated inside the code, so no external image file is needed.
