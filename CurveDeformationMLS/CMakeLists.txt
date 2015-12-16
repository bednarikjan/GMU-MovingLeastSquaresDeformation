cmake_minimum_required (VERSION 2.6) 
project (ShapeDeformation) 

set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

include(cotire)

add_definitions("-g")

find_package(OpenCV 2.4 REQUIRED)

find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIR})

file(GLOB SOURCES ./*.cpp)

add_executable(ShapeDeformation
    ${SOURCES}
)
	
target_link_libraries(ShapeDeformation
	${OpenCV_LIBS}
        ${OPENCL_LIBRARIES}
#        mgl
	)
	
set_target_properties(ShapeDeformation PROPERTIES COTIRE_CXX_PREFIX_HEADER_INIT "std.h")
#set_target_properties(ShapeDeformation PROPERTIES COMPILE_FLAGS "-DWITHOUT_OPENCL -DHAVE_MATHGL")
set_target_properties(ShapeDeformation PROPERTIES COMPILE_FLAGS "-DWITHOUT_OPENCL")

cotire(ShapeDeformation)
