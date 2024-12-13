#include "jubjub/eddsa.hpp"
#include "utils.hpp"


/*
To generate random signatures for testing:

>>> from ethsnarks.eddsa import * 
... k, A = EdDSA.random_keypair() 
... A, sig, m = EdDSA.sign(b'abcd', k)  
*/

/*
Test-vectors for default parameters with known key

from ethsnarks.eddsa import *
k = FQ(92734322978479831564281963181193415354487363923837807727447121691861920913223)
pmsg = pureeddsa_sign(b'abcd', k)
print(pmsg.A, pmsg.sig.R, pmsg.sig.s)
emsg = eddsa_sign(b'abc', k)
print(emsg.A, emsg.sig.R, emsg.sig.s)
*/

int main( int argc, char **argv )
{
    ppT::init_public_params();

    Params params;

    const char *msg_abcd = "abcd";
    const auto msg_abcd_bits = bytes_to_bv((const uint8_t*)msg_abcd, strlen(msg_abcd));

    const char *msg_abc = "abc";
    const auto msg_abc_bits = bytes_to_bv((const uint8_t*)msg_abc, strlen(msg_abc));

    const EdwardsPoint A(FieldT("333671881179914989291633188949569309119725676183802886621140166987382124337"),
                         FieldT("4050436616325076046600891135828313078248584449767955905006778857958871314574"));

    // Verify HashEdDSA - where message is hashed prior to signing    
    if( ! eddsa_open<EdDSA>(
        params, A, {
            {
                FieldT("21473010389772475573783051334263374448039981396476357164143587141689900886674"),
                FieldT("11330590229113935667895133446882512506792533479705847316689101265088791098646")
            },
            FieldT("21807294168737929637405719327036335125520717961882955117047593281820367379946")
        },
        msg_abc_bits
    )) {
        std::cerr << "FAIL HashEdDSA\n";
        return 1;
    }

    // Verify PureEdDSA where no message compression is used for H(R,A,M)
    if( ! eddsa_open<PureEdDSA>(
        params, A, {
            {
                FieldT("17815983127755465894346158776246779862712623073638768513395595796132990361464"),
                FieldT("947174453624106321442736396890323086851143728754269151257776508699019857364")
            },
            FieldT("13341814865473145800030207090487687417599620847405735706082771659861699337012")
        },
        msg_abcd_bits
    )) {
        std::cerr << "FAIL PureEdDSA\n";
        return 1;
    }
    
    
    const EdwardsPoint AA(FieldT("13866189866966177725545955315473288366756005698510089107323854294421070766415"),
                         FieldT("5678312344112160219869310884883423207452282246972904837512646127791201675101"));

    // Verify PureEdDSA where no message compression is used for H(R,A,M)
    if( ! eddsa_open<PureEdDSA>(
        params, AA, {
            {
                FieldT("4087183505664342292180507405613085578529509817727156483742798078248408107275"),
                FieldT("20404999006901888240286251079916813811703391280366803120533036358698013630377")
            },
            FieldT("11057626389626008869885381487301595944953769128610323633054704395704413883275")
        },
        msg_abcd_bits
    )) {
        std::cerr << "FAIL My PureEdDSA\n";
        return 1;
    }
    std::cout << "OK\n";
    return 0;
}
