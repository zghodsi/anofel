#include "jubjub/anofel_setup.hpp"
#include "utils.hpp"
#include "stubs.hpp"


struct tree_things {

	VariableArrayT address_bits;
	VariableArrayT path;
	VariableT leaf;
	VariableT expected_root;

};

tree_things make_tree(ProtoboardT& pb, size_t tree_depth)
{
	if (tree_depth < 4 || tree_depth > 9)
	{
		std::cerr << "Tree depth not valid" << std::endl;
	}

	// leaf = mimc([A.x, A.y],1)
	tree_things Tree;
    FieldT is_right = 1;
	
	Tree.address_bits.allocate(pb, tree_depth, "address_bits");
	Tree.path.allocate(pb, tree_depth, "path");
	Tree.leaf.allocate(pb, "leaf");
	Tree.expected_root.allocate(pb, "expected_root");

	FieldT right0 = FieldT("15204944786894089078759014777674833662563213958500200405269309905769130586874");
	std::vector<FieldT> left = {
		FieldT("3703141493535563179657531719960160174296085208671919316200479060314459804651"),
		FieldT("11714008893116939441510788599557636816518527327543193374630310875272509334396"),
		FieldT("9881790034808292405036271961589462686158587796044671417688221824074647491645"),
		FieldT("11437467823393790387399137249441941313717686441929791910070352316474327319704"),
		FieldT("918403109389145570117360101535982733651217667914747213867238065296420114726"),
		FieldT("3478732947298347927183478374910478379274104719749173491741074107438921394719"),
		FieldT("74987571106198374819616304519818748279508198761849302850485197817687263940580"),
		FieldT("93846615369450037265176438297409274814876861768736498294023840284274186180052"),
		FieldT("78372971971080347928374917619691948038527316381639470845023749181629174982054"),
		FieldT("78324924701038577602437390577378123235008212159332933741981927394038401817350")};

	std::vector<FieldT> root = {
		FieldT(""),
		FieldT(""),
		FieldT(""),
		FieldT(""),
		FieldT("15290904905826984033526879532396396567891715029697887378538510176885078324430"), //4
		FieldT("16730802325196002280779549731060732012066610271362595765418761346975868638479"),
		FieldT("19981859684259418841244698846896506098929365709019198417141771515947900412698"),
		FieldT("2189600103516413824130076660508327756104855793429359658595146882354520478386"),
		FieldT("11314194192463869605814120000017399512517991671029665353028015813785017107000"),
		FieldT("18694986038725275382254448895114571446093715281094762453157396575625120003299")};

	for (int i=0; i<tree_depth; i++){
  		pb.val(Tree.address_bits[i]) = is_right;
	}

	for (int i=0; i<tree_depth; i++){
  		pb.val(Tree.path[i]) = left[i];
	}
    
    pb.val(Tree.leaf) = right0;
    pb.val(Tree.expected_root) = root[tree_depth];


	// depth 1
	//auto left = FieldT("3703141493535563179657531719960160174296085208671919316200479060314459804651");
	//auto right = FieldT("15204944786894089078759014777674833662563213958500200405269309905769130586874");
	//auto root = FieldT("8780268166819098560340482100794469656734976947836332645457046631849839412225");
	//
	//pb.val(Tree.address_bits[0]) = is_right;
	//pb.val(Tree.path[0]) = left;
	//pb.val(Tree.leaf) = right;
	//pb.val(Tree.expected_root) = root;

	return Tree;
}


bool test_anofel_setup()
{
	Params params;
	ProtoboardT pb;

	// H(D) and salt
	VariableArrayT D;
	D.allocate(pb, 256, "D");
	D.fill_with_bits_of_field_element(pb, 
			FieldT("16838670147829712932420991684129000253378636928981731224589534936353716235035"));
	VariableArrayT salt;
	salt.allocate(pb, 256, "salt");
	salt.fill_with_bits_of_field_element(pb, 
			FieldT("4937932098257800452675892262662102197939919307515526854605530277406221704113"));

	// expected comm = H(mpk.x||mpk.y||H(D)||salt)
	EdwardsPoint comm={FieldT("3563683065437001431022117700507934918711183407057576880270573744331195180154"),
                        FieldT("3131904386149043174130186150300263619184691479586087897758820638841563706312")};

	// client (msk, mpk)
	// pub key
	const EdwardsPoint mpk(FieldT("13866189866966177725545955315473288366756005698510089107323854294421070766415"),
			FieldT("5678312344112160219869310884883423207452282246972904837512646127791201675101"));

	// secret key
	VariableArrayT msk;
	msk.allocate(pb, 256, "msk");
	msk.fill_with_bits_of_field_element(pb, 
			FieldT("464113509201360387371747793490702926537166176955469493894948758186401155582"));

	// signature over H(D)||mpk with PKc
	// PKc
	const EdwardsPoint A(FieldT("13028030416059925829793459802476463047496081151168513807594046867053955618897"),
			FieldT("21690986651052178259365509859844306381913248886471329937412829414390485618932"));
	// signature (R,S) 
	const EdwardsPoint R(FieldT("13857209869959987729875264559554629861579137238241224160053473964122321399643"),
			FieldT("11183935187923421522362386808762601229986188266691479765642561376734292385501"));
	VariableArrayT sig;
	sig.allocate(pb, 256, "sig");
	sig.fill_with_bits_of_field_element(pb, 
			FieldT("21453576121563227919689532221154311940737355146082409111653604831111017053713"));

	// merkle tree
	size_t tree_depth = 4;
	tree_things Tree = make_tree(pb, tree_depth);


	// client (sk_sig, pk_sig)
	// sig pub key
	const EdwardsPoint pk_sig(FieldT("5083434580149958055596800160661984003386576121518647827517546039984835648342"),
			FieldT("11960404361972272492756306028799358420822926508162738013772618584356357981426"));

	// sig secret key
	VariableArrayT sk_sig;
	sk_sig.allocate(pb, 256, "sk_sig");
	sk_sig.fill_with_bits_of_field_element(pb, 
			FieldT("6453482891510615431577168724743356132495662554103773572771861111634748265227"));

	// expected tag = H(msk||H(pk_sig))
	EdwardsPoint tag={FieldT("8488396915141738869228447857914335369306195367929216603698283782314665107548"),
                        FieldT("2712301624239834090213547548562992599618685801372774704694386351465833685307")};


	anofel_setup the_gadget(pb, params, D, salt, msk, mpk, comm, 
			A, R, sig, sk_sig, pk_sig, tag, 
			tree_depth, Tree.address_bits, merkle_tree_IVs(pb), Tree.leaf, Tree.expected_root, Tree.path,
			//tree_depth, address_bits, merkle_tree_IVs(pb), leaf, expected_root, path,
			"the_gadget");

	// set the first 10 inputs as public
	//pb.set_input_sizes(50);

	the_gadget.generate_r1cs_witness();
	the_gadget.generate_r1cs_constraints();

	if (comm.x != pb.val(the_gadget.comm_x()))
	{
		std::cerr << "FAIL unexpected comm_x" << std::endl;
		std::cerr << "Expected:"; comm.x.print();
		std::cerr << "  Actual:"; pb.val(the_gadget.comm_x()).print();
	}
	if (comm.y != pb.val(the_gadget.comm_y()))
	{
		std::cerr << "FAIL unexpected comm_y" << std::endl;
		std::cerr << "Expected:"; comm.y.print();
		std::cerr << "  Actual:"; pb.val(the_gadget.comm_y()).print();
	}
	
	if (tag.x != pb.val(the_gadget.tag_x()))
	{
		std::cerr << "FAIL unexpected tag_x" << std::endl;
		std::cerr << "Expected:"; tag.x.print();
		std::cerr << "  Actual:"; pb.val(the_gadget.tag_x()).print();
	}
	if (tag.y != pb.val(the_gadget.tag_y()))
	{
		std::cerr << "FAIL unexpected tag_y" << std::endl;
		std::cerr << "Expected:"; tag.y.print();
		std::cerr << "  Actual:"; pb.val(the_gadget.tag_y()).print();
	}


    std::cout << pb.num_constraints() << " constraints" << std::endl;

    if( ! pb.is_satisfied() ) {
        std::cerr << "Not satisfied!" << std::endl;
        return false;
    }

	//return pb.is_satisfied();
	return stub_test_proof_verify(pb);
}


int main( void )
{
	ppT::init_public_params();

	if( ! test_anofel_setup() )
	{
		std::cerr << "FAIL\n";
		return 1;
	}

	std::cout << "OK\n";
	return 0;
}
