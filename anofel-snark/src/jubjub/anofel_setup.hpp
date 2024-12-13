#ifndef ANOFEL_SETUP_
#define ANOFEL_SETUP_

#include "jubjub/params.hpp"
#include "jubjub/pedersen_hash.hpp"
#include "jubjub/pub_key.hpp"

#include "gadgets/field2bits_strict.hpp"
#include "jubjub/eddsa.hpp"

#include "gadgets/merkle_tree.hpp"
#include "gadgets/mimc.hpp"


class anofel_setup : public GadgetT
{
	public:
		field2bits_strict m_mpk_x_bits;
		field2bits_strict m_mpk_y_bits;
		
		field2bits_strict m_pksig_x_bits;
		field2bits_strict m_pksig_y_bits;

		const VariableArrayT m_mpkDsalt;
		const VariableArrayT m_mpkD;
		const VariableArrayT m_pksig;
		PedersenHash m_hash_commit;
		EdDSA_PubKey m_pubkey;
		PureEdDSA m_eddsa;
		merkle_path_authenticator<MiMC_e7_hash_gadget> m_merkle;
		MiMC_e7_hash_gadget m_mimc;
		PedersenHashToBits m_hash_pksig;
		PedersenHash m_hash_tag;

		const EdwardsPoint m_mpk;
		const EdwardsPoint m_comm;
		const EdwardsPoint m_tag;
		const VariableT m_mimc_A;

		anofel_setup(
				ProtoboardT& in_pb,
				const Params& in_params,
				const VariableArrayT& in_D,
				const VariableArrayT& in_salt,
				const VariableArrayT& in_msk,
				const EdwardsPoint& in_mpk,
				const EdwardsPoint& in_comm,
				const EdwardsPoint& in_A,
				const EdwardsPoint& in_R,
				const VariableArrayT& in_s,
				const VariableArrayT& in_sk_sig,
				const EdwardsPoint& in_pk_sig,
				const EdwardsPoint& in_tag,
				const size_t in_depth,
				const VariableArrayT in_address_bits,
				const VariableArrayT in_IVs,
				const VariableT in_leaf,
				const VariableT in_expected_root,
				const VariableArrayT in_path,
				const std::string& annotation_prefix
				);

		void generate_r1cs_constraints();

		void generate_r1cs_witness();

		const VariableT& comm_x();
		const VariableT& comm_y();

		const VariableT& tag_x();
		const VariableT& tag_y();

};


// ANOFEL_SETUP_
#endif
