add_executable( cpa_test cpa_test.cpp)
target_link_libraries(cpa_test PRIVATE resreach3)

add_executable(dpa_test dpa_test.cpp)
target_link_libraries(dpa_test PRIVATE resreach3)

add_executable(spa_test spa_test.cpp)
target_link_libraries(spa_test PRIVATE resreach3)

add_executable(hpa_test hpa_test.cpp)
target_link_libraries(hpa_test PRIVATE resreach3)

add_executable(z_functions_test z_functions_test.cpp)
target_link_libraries(z_functions_test PRIVATE resreach3)