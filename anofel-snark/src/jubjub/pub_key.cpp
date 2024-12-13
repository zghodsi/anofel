#include "jubjub/pub_key.hpp"
#include "utils.hpp"

EdDSA_PubKey::EdDSA_PubKey(
    ProtoboardT& in_pb,
    const Params& in_params,
    const VariableArrayT& in_k,
    const std::string& annotation_prefix
) :
    GadgetT(in_pb, annotation_prefix),
    m_mul(in_pb, in_params, in_params.Gx, in_params.Gy, in_k, FMT(this->annotation_prefix, ".mul"))
{ }


void EdDSA_PubKey::generate_r1cs_constraints()
{
    m_mul.generate_r1cs_constraints();
}

void EdDSA_PubKey::generate_r1cs_witness()
{
    m_mul.generate_r1cs_witness();
}

const VariableT& EdDSA_PubKey::result_x() const
{
    return m_mul.result_x();
}

const VariableT& EdDSA_PubKey::result_y() const
{
    return m_mul.result_y();
}


