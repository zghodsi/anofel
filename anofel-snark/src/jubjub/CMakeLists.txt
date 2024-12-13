#include_directories(.)

add_library(jubjub STATIC
    fixed_base_mul.cpp
    conditional_point.cpp
    scalarmult.cpp
    adder.cpp
    doubler.cpp
    isoncurve.cpp
    commitment.cpp
    notloworder.cpp
    validator.cpp
    point.cpp
    pedersen_hash.cpp
    fixed_base_mul_zcash.cpp
    montgomery.cpp
    eddsa.cpp
    pub_key.cpp
    anofel_setup.cpp
    anofel_training.cpp
    )

target_include_directories(jubjub PUBLIC
  ${DEPENDS_DIR}/libsnark
  ${DEPENDS_DIR}/libsnark/depends/libfqfft
  ${DEPENDS_DIR}/libsnark/depends/libff)

target_link_libraries(jubjub gadgets)
