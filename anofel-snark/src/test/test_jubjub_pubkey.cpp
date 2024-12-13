#include "jubjub/pub_key.hpp"
#include "jubjub/pedersen_hash.hpp"
#include "utils.hpp"

bool test_jubjub_pubkey(const FieldT& sk, size_t sk_size, const EdwardsPoint& expected)
{
    const Params params;
    ProtoboardT pb;

    VariableArrayT k;
    k.allocate(pb, sk_size, "k");
    k.fill_with_bits_of_field_element(pb, sk);

    EdDSA_PubKey the_gadget(pb, params, k, "the_gadget");
    the_gadget.generate_r1cs_constraints();
    the_gadget.generate_r1cs_witness();

    
    bool is_ok = true;
    if ( expected.x != pb.val(the_gadget.result_x()) )
    {
        std::cerr << "FAIL unexpected" << std::endl;
		std::cerr << "Expected:"; expected.x.print();
		std::cerr << "  Actual:"; pb.val(the_gadget.result_x()).print();
		is_ok = false;
	}

    if ( expected.y != pb.val(the_gadget.result_y()) )
    {
        std::cerr << "FAIL unexpected" << std::endl;
		std::cerr << "Expected:"; expected.y.print();
		std::cerr << "  Actual:"; pb.val(the_gadget.result_y()).print();
		is_ok = false;
	}
    
    return is_ok && pb.is_satisfied();
}

int main(void)
{
    ppT::init_public_params();
    // secret key
    FieldT sk("464113509201360387371747793490702926537166176955469493894948758186401155582");
    // pub key
    EdwardsPoint expected = {
        FieldT("13866189866966177725545955315473288366756005698510089107323854294421070766415"),
        FieldT("5678312344112160219869310884883423207452282246972904837512646127791201675101")};
    if (!test_jubjub_pubkey(sk, 252, expected))
    {
        std::cerr << "FAIL\n";
        return 1;
    }

    std::cout << "OK\n";
    return 0;
}
