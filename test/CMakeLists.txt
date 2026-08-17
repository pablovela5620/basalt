cmake_minimum_required(VERSION 3.10...3.18)

# Note: add_subdirectory(googletest ...) is called in basalt-headers

include_directories(../thirdparty/basalt-headers/test/include)

add_executable(test_spline_opt src/test_spline_opt.cpp)
target_link_libraries(test_spline_opt gtest gtest_main basalt_internal)

add_executable(test_vio src/test_vio.cpp)
target_link_libraries(test_vio gtest gtest_main basalt_internal)

add_executable(test_nfr src/test_nfr.cpp)
target_link_libraries(test_nfr gtest gtest_main basalt_internal)

add_executable(test_qr src/test_qr.cpp)
target_link_libraries(test_qr gtest gtest_main basalt_internal)

add_executable(test_linearization src/test_linearization.cpp)
target_link_libraries(test_linearization gtest gtest_main basalt_internal)

add_executable(test_patch src/test_patch.cpp)
target_link_libraries(test_patch gtest gtest_main basalt_internal)

if(BASALT_ENABLE_RERUN)
  add_executable(test_rerun_vit_adapter src/test_rerun_vit_adapter.cpp)
  target_link_libraries(test_rerun_vit_adapter gtest gtest_main basalt_rerun)

  add_executable(test_rerun_offline_adapter src/test_rerun_offline_adapter.cpp)
  target_link_libraries(test_rerun_offline_adapter gtest gtest_main basalt_rerun)
endif()

if(BASALT_ENABLE_ROBOCAP)
  add_executable(test_robocap_dataset src/test_robocap_dataset.cpp)
  target_link_libraries(test_robocap_dataset gtest gtest_main basalt_internal SQLite::SQLite3)
endif()

enable_testing()

include(GoogleTest)

gtest_add_tests(TARGET test_spline_opt AUTO)
gtest_add_tests(TARGET test_vio AUTO)
gtest_add_tests(TARGET test_nfr AUTO)
gtest_add_tests(TARGET test_qr AUTO)
gtest_add_tests(TARGET test_linearization AUTO)
gtest_add_tests(TARGET test_patch AUTO)
if(BASALT_ENABLE_RERUN)
  gtest_add_tests(TARGET test_rerun_vit_adapter AUTO)
  gtest_add_tests(TARGET test_rerun_offline_adapter AUTO)
endif()
if(BASALT_ENABLE_ROBOCAP)
  gtest_add_tests(TARGET test_robocap_dataset AUTO)
endif()
