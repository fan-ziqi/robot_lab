cmake_minimum_required(VERSION 3.8)
project(t1_description)

find_package(ament_cmake REQUIRED)

install(
        DIRECTORY meshes launch urdf config xacro
        DESTINATION share/${PROJECT_NAME}/
)

ament_package()