#include "stubs.hpp"

bool stub_test_proof_verify( const ProtoboardT &in_pb )
{
    auto constraints = in_pb.get_constraint_system();
    auto keypair = libsnark::r1cs_gg_ppzksnark_generator<ppT>(constraints);
    //auto keypair = libsnark::r1cs_se_ppzksnark_generator<ppT>(constraints);

    auto primary_input = in_pb.primary_input();
    auto auxiliary_input = in_pb.auxiliary_input();
	std::cout << "Primary (public) input size: " << in_pb.num_inputs() << std::endl;
	//std::cout << "Primary (public) input: " << primary_input << std::endl;
    //std::cout << "Auxiliary (private) input: " << auxiliary_input << std::endl;
	Benchmark bench_p;
    auto proof = libsnark::r1cs_gg_ppzksnark_prover<ppT>(keypair.pk, primary_input, auxiliary_input);
    //auto proof = libsnark::r1cs_se_ppzksnark_prover<ppT>(keypair.pk, primary_input, auxiliary_input);
	bench_p.print("prove");
	Benchmark bench_v;
	bool verified = libsnark::r1cs_gg_ppzksnark_verifier_strong_IC <ppT> (keypair.vk, primary_input, proof);
	//bool verified = libsnark::r1cs_se_ppzksnark_verifier_strong_IC <ppT> (keypair.vk, primary_input, proof);
	bench_v.print("verify");

    return verified;
}
