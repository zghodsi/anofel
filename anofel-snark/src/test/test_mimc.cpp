#include "utils.hpp"
#include "gadgets/mimc.hpp"

//using ethsnarks::ppT;
//using ethsnarks::FieldT;
//using ethsnarks::ProtoboardT;
//using ethsnarks::VariableT;
//using ethsnarks::MiMC_e7_gadget;
//using ethsnarks::make_variable;


struct MiMC_TestCase
{
    FieldT plaintext;
    FieldT key;
    FieldT result;
};


bool test_MiMC(const MiMC_TestCase& test_case)
{
    ProtoboardT pb;

    const VariableT in_x = make_variable(pb, test_case.plaintext, "x");
    const VariableT in_k = make_variable(pb, test_case.key, "k");
    pb.set_input_sizes(2);

    MiMC_e7_gadget the_gadget(pb, in_x, in_k, "the_gadget");
    the_gadget.generate_r1cs_witness();
    the_gadget.generate_r1cs_constraints();

    if( test_case.result != pb.val(the_gadget.result()) )
    {
        std::cerr << "Unexpected result!\n";
        return false;
    }

    std::cout << pb.num_constraints() << " constraints" << std::endl;

    return pb.is_satisfied();
}


int main( int argc, char **argv )
{
    ppT::init_public_params();

    const std::vector<MiMC_TestCase> test_cases = {
        {FieldT("3703141493535563179657531719960160174296085208671919316200479060314459804651"),
         FieldT("134551314051432487569247388144051420116740427803855572138106146683954151557"),
         FieldT("11437467823393790387399137249441941313717686441929791910070352316474327319704")}
    };

    int i = 0;
    for( const auto& tc : test_cases )
    {
        if( ! test_MiMC(tc) )
        {
            std::cerr << "FAIL " << i << std::endl;
            return 1;
        }

        i += 1;
    }

    std::cout << "OK" << std::endl;
    return 0;
}

