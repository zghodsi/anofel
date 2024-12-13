#ifndef JUBJUB_ISONCURVE_HPP_
#define JUBJUB_ISONCURVE_HPP_

// Copyright (c) 2018 @HarryR
// License: LGPL-3.0+

#include "jubjub/params.hpp"



/**
* Validates that the x/y coordinates are on the curve
*
* a*x*x + y*y = 1 + d*x*x*y*y
*/
class IsOnCurve : public GadgetT {
public:
	const Params& m_params;
	const VariableT m_X;
	const VariableT m_Y;
	const VariableT m_xx;
	const VariableT m_yy;

	IsOnCurve(
        ProtoboardT& in_pb,
        const Params& in_params,
        const VariableT in_X,
        const VariableT in_Y,
        const std::string& annotation_prefix
    );

    void generate_r1cs_constraints();

    void generate_r1cs_witness();
};



// JUBJUB_ISONCURVE_HPP_
#endif
