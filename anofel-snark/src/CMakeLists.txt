include_directories(.)

add_library(common STATIC utils.cpp stubs.cpp crypto/sha256.c)
target_link_libraries(common ff ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  ${DEPENDS_DIR}/libsnark
  ${DEPENDS_DIR}/libsnark/depends/libfqfft
  ${DEPENDS_DIR}/libsnark/depends/libff)

add_subdirectory(gadgets)
add_subdirectory(jubjub)
add_subdirectory(test)

