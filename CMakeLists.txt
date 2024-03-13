cmake_minimum_required(VERSION 3.10)
project(KNN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt5 COMPONENTS Widgets REQUIRED)

if(Qt5Widgets_FOUND)
  message(STATUS "Qt5 package found")
else()
  message(FATAL_ERROR "Qt5 package not found")
endif()

add_executable(KNN main.cpp)

target_link_libraries(KNN Qt5::Widgets stdc++fs)

option(HAVE_DEBUG "Build with debug information" OFF)

if(HAVE_DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")