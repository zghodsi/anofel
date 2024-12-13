add_subdirectory(libsnark)
set(DEPENDS_DIR_SHA3IUF "${DEPENDS_DIR}/SHA3IUF/")

add_library(SHA3IUF STATIC ${DEPENDS_DIR_SHA3IUF}/sha3.c)
target_include_directories(SHA3IUF PUBLIC ${DEPENDS_DIR_SHA3IUF})
target_compile_definitions(SHA3IUF PRIVATE SHA3_USE_KECCAK=1)

add_executable(sha3sum ${DEPENDS_DIR_SHA3IUF}/sha3sum.c)
target_link_libraries(sha3sum SHA3IUF)

add_executable(sha3test ${DEPENDS_DIR_SHA3IUF}/sha3test.c)
target_link_libraries(sha3test SHA3IUF)
add_test(NAME run_sha3test COMMAND sha3test)
