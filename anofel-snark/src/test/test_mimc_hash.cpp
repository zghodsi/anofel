#include "gadgets/mimc.hpp"
#include "stubs.hpp"



bool test_mimc_hash()
{
    ProtoboardT pb;

    // Public inputs
    VariableT m_0 = make_variable(pb, FieldT("3703141493535563179657531719960160174296085208671919316200479060314459804651"), "m_0");
    VariableT m_1 = make_variable(pb, FieldT("134551314051432487569247388144051420116740427803855572138106146683954151557"), "m_1");
    pb.set_input_sizes(2);

    // Private inputs
    VariableT iv = make_variable(pb, FieldT("918403109389145570117360101535982733651217667914747213867238065296420114726"), "iv");

    MiMC_e7_hash_gadget the_gadget(pb, iv, {m_0, m_1}, "gadget");
    the_gadget.generate_r1cs_witness();
    the_gadget.generate_r1cs_constraints();

    auto result_expected = FieldT("15683951496311901749339509118960676303290224812129752890706581988986633412003");
    if( result_expected != pb.val(the_gadget.result()) )
    {
        std::cerr << "Unexpected result!\n";

        std::cerr << "Expected: ";
        result_expected.print();

        std::cerr << "Cipher Output: ";
        pb.val(the_gadget.m_ciphers.back().result()).print();

        std::cerr << "Result: ";
        pb.val(the_gadget.result()).print();

        std::cerr << "Outputs:" << std::endl;
        int i = 0;
        for( auto& x : the_gadget.m_outputs )
        {
            std::cerr << i << " = ";
            pb.val(x).print();
            i += 1;
        }

        std::cerr << "Messages:" << std::endl;
        i = 0;
        for( auto& x : the_gadget.m_messages )
        {
            std::cerr << i << " = ";
            pb.val(x).print();
            i += 1;
        }

        return false;
    }

    std::cout << pb.num_constraints() << " constraints" << std::endl;

    if( ! pb.is_satisfied() ) {
        std::cerr << "Not satisfied!" << std::endl;
        return false;
    }

    return stub_test_proof_verify(pb);
}




int main( int argc, char **argv )
{
    // Types for board
    ppT::init_public_params();

    if( ! test_mimc_hash() )
    {
        std::cerr << "FAIL\n";
        return 1;
    }

    std::cout << "OK\n";
    return 0;
}

