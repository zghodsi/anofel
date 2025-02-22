// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#include "jubjub/validator.hpp"


PointValidator::PointValidator(
    ProtoboardT& in_pb,
    const Params& in_params,
    const VariableT in_X,
    const VariableT in_Y,
    const std::string& annotation_prefix
) :
	GadgetT(in_pb, annotation_prefix),
	m_notloworder(in_pb, in_params, in_X, in_Y, FMT(this->annotation_prefix, ".notloworder")),
	m_isoncurve(in_pb, in_params, in_X, in_Y, FMT(this->annotation_prefix, ".isoncurve"))
{

}


void PointValidator::generate_r1cs_constraints()
{
	m_isoncurve.generate_r1cs_constraints();
	m_notloworder.generate_r1cs_constraints();
}


void PointValidator::generate_r1cs_witness()
{
	m_isoncurve.generate_r1cs_witness();
	m_notloworder.generate_r1cs_witness();
}


