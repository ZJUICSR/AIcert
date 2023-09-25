add_executable( cpa_test cpa_test.cpp)
target_link_libraries(cpa_test PRIVATE resreach3)

add_executable(dpa_test dpa_test.cpp)
target_link_libraries(dpa_test PRIVATE resreach3)

add_executable(hpa_test hpa_test.cpp)
target_link_libraries(hpa_test PRIVATE resreach3)

add_executable(z_functions_test z_functions_test.cpp)
target_link_libraries(z_functions_test PRIVATE resreach3)

add_executable(ttest_test ttest_test.cpp)
target_link_libraries(ttest_test PRIVATE resreach3)

add_executable(x2test_test x2test_test.cpp)
target_link_libraries(x2test_test PRIVATE resreach3)

find_package(Threads REQUIRED)
set(CMAKE_CXX_FLAGS "${CAMKE_CXX_FLAGS} -std=c++11 -pthread")

add_executable(spa_test spa_test.cpp)
target_link_libraries(spa_test PRIVATE resreach3)
# set_target_properties(spa_test PROPERTIES CXX_STANDARD 11
# COMPILE_FLAGS "-pthread"
# LINK_FLAGS "-pthread")

# find_library(r3_lib NAMES resreach3 PATHS ../)

# #so
# add_executable(cpa_test cpa_test.cpp)
# target_link_libraries(cpa_test PRIVATE ${r3_lib})

# add_executable(dpa_test dpa_test.cpp)
# target_link_libraries(dpa_test PRIVATE ${r3_lib})

# add_executable(spa_test spa_test.cpp)
# target_link_libraries(spa_test PRIVATE ${r3_lib})

# add_executable(hpa_test hpa_test.cpp)
# target_link_libraries(hpa_test PRIVATE ${r3_lib})

# add_executable(z_functions_test z_functions_test.cpp)
# target_link_libraries(z_functions_test PRIVATE ${r3_lib})

# add_executable(ttest_test ttest_test.cpp)
# target_link_libraries(ttest_test PRIVATE ${r3_lib})

# add_executable(x2test_test x2test_test.cpp)
# target_link_libraries(x2test_test PRIVATE ${r3_lib})