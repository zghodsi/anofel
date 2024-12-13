#ifndef JUBJUB_PUB_KEY_
#define JUBJUB_PUB_KEY_

#include "types.hpp"

#include "jubjub/params.hpp"
#include "pedersen_hash.hpp"
#include "fixed_base_mul.hpp"


class EdDSA_PubKey : public GadgetT
{
public:
    fixed_base_mul m_mul;

    EdDSA_PubKey(
        ProtoboardT& in_pb,
        const Params& in_params,
        const VariableArrayT& in_k,
        const std::string& annotation_prefix
    );

    void generate_r1cs_constraints();

    void generate_r1cs_witness();

    const VariableT& result_x() const;
    const VariableT& result_y() const;
};


// JUBJUB_PUB_KEY_
#endif
