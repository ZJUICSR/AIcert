# cmake_minimum_required(VERSION 3.0)
# project(NVWA)

# include_directories(inc)

# add_subdirectory(src)
# add_subdirectory(tests)
# add_subdirectory(bench)
# add_subdirectory(examples)


cmake_minimum_required(VERSION 3.0)
project(resreach3)

include_directories(Inc)

add_library(resreach3 SHARED Src/dummy.cpp)#SHARED

aux_source_directory(Src/AttackMethods ATTACKMETHODS_SRC)
target_sources(resreach3 PRIVATE ${ATTACKMETHODS_SRC})

aux_source_directory(Src/BASETOOL BASETOOL_SRC)
target_sources(resreach3 PRIVATE ${BASETOOL_SRC})

aux_source_directory(Src/CNNModel CNNMODEL_SRC)
target_sources(resreach3 PRIVATE ${CNNMODEL_SRC})

aux_source_directory(Src/TRACE TRACE_SRC)
target_sources(resreach3 PRIVATE ${TRACE_SRC})

add_subdirectory(Examples)
add_subdirectory(Tests)



