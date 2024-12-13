file(GLOB gadget_sources "*.cpp")
add_library(gadgets STATIC ${gadget_sources})

target_link_libraries(gadgets common SHA3IUF)

target_include_directories(gadgets PUBLIC
  ${DEPENDS_DIR}/libsnark
  ${DEPENDS_DIR}/libsnark/depends/libfqfft
  ${DEPENDS_DIR}/libsnark/depends/libff)
