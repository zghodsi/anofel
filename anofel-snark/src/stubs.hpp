#ifndef STUBS_HPP
#define STUBS_HPP
#include <libsnark/zk_proof_systems/ppzksnark/r1cs_gg_ppzksnark/r1cs_gg_ppzksnark.hpp>
//#include <libsnark/zk_proof_systems/ppzksnark/r1cs_se_ppzksnark/r1cs_se_ppzksnark.hpp>
#include "types.hpp"
#include <chrono>

class Benchmark {
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
public:
    void print(std::string action) {
        auto dur = std::chrono::steady_clock::now() - begin;
		std::cerr << "ZKPROOF_BENCHMARK: {"
             << R"("action": ")" << action << "\", "
             << R"("microseconds": )"
             << std::chrono::duration_cast<std::chrono::microseconds>(dur).count()
             << "}" << std::endl;
    }
};

bool stub_test_proof_verify( const ProtoboardT &in_pb );

#endif
