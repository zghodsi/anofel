#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <libff/algebra/curves/alt_bn128/alt_bn128_pp.hpp>
#include <libsnark/gadgetlib1/protoboard.hpp>
#include <libsnark/gadgetlib1/gadget.hpp>


typedef libff::bigint<libff::alt_bn128_r_limbs> LimbT;
typedef libff::alt_bn128_G1 G1T;
typedef libff::alt_bn128_G2 G2T;
typedef libff::alt_bn128_pp ppT;
typedef libff::Fq<ppT> FqT;
typedef libff::Fr<ppT> FieldT;
typedef libsnark::r1cs_constraint<FieldT> ConstraintT;
typedef libsnark::protoboard<FieldT> ProtoboardT;
typedef libsnark::pb_variable<FieldT> VariableT;
typedef libsnark::pb_variable_array<FieldT> VariableArrayT;
typedef libsnark::pb_linear_combination<FieldT> LinearCombinationT;
typedef libsnark::pb_linear_combination_array<FieldT> LinearCombinationArrayT;
typedef libsnark::linear_term<FieldT> LinearTermT;
typedef libsnark::gadget<FieldT> GadgetT;


#endif
