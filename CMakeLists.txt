CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(demo1)

find_package(OpenCV REQUIRED)

ADD_EXECUTABLE(demo1  main.cpp)

target_link_libraries(demo1 PRIVATE ${OpenCV_LIBS})
