#include "jubjub/anofel_training.hpp"
#include "utils.hpp"


anofel_training::anofel_training(
		ProtoboardT& in_pb,
		const Params& in_params,
		//const VariableArrayT& in_D,
		//const VariableArrayT& in_salt,
		const VariableArrayT& in_msk,
		//const EdwardsPoint& in_mpk,
		//const EdwardsPoint& in_comm,
		//const EdwardsPoint& in_A,
		//const EdwardsPoint& in_R,
		//const VariableArrayT& in_s,
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
		) :
	GadgetT(in_pb, annotation_prefix), m_tag(in_tag),
		//m_mpk(in_mpk), m_comm(in_comm), m_mimc_A(in_leaf), 

	//m_mpk_x_bits(in_pb, in_mpk.as_VariablePointT(in_pb, "mpk").x, FMT(this->annotation_prefix, ".mpk_x_bits")),
	//m_mpk_y_bits(in_pb, in_mpk.as_VariablePointT(in_pb, "mpk").y, FMT(this->annotation_prefix, ".mpk_y_bits")),

	//m_mpkDsalt(flatten({
	//		m_mpk_x_bits.result(),
	//		m_mpk_y_bits.result(),
	//		in_D,
	//		in_salt,
	//		})),
	//m_mpkD(flatten({
	//		m_mpk_x_bits.result(),
	//		m_mpk_y_bits.result(),
	//		in_D,
	//		})),

	m_pksig_x_bits(in_pb, in_pk_sig.as_VariablePointT(in_pb, "pksig").x, 
					FMT(this->annotation_prefix, ".pksig_x_bits")),
	m_pksig_y_bits(in_pb, in_pk_sig.as_VariablePointT(in_pb, "pksig").y, 
					FMT(this->annotation_prefix, ".pksig_y_bits")),
	m_pksig(flatten({
			m_pksig_x_bits.result(),
			m_pksig_y_bits.result(),
			})),
	
	//m_hash_commit(in_pb, in_params, "Dsalt_commit", m_mpkDsalt, FMT(this->annotation_prefix, ".Dsalt_hash")),
	//m_pubkey(in_pb, in_params, in_msk, FMT(this->annotation_prefix, ".pubkey")),
	//m_eddsa(in_pb, in_params, EdwardsPoint(in_params.Gx, in_params.Gy), in_A.as_VariablePointT(in_pb, "A"), 
	//		in_R.as_VariablePointT(in_pb, "R"), in_s, in_D, ".eddsa"),
	//m_mimc(in_pb, make_variable(pb, FieldT::one(), "iv"), {in_A.as_VariablePointT(in_pb, "A").x, in_A.as_VariablePointT(in_pb, "A").y}, "mimc"),
	m_merkle(in_pb, in_depth, in_address_bits, in_IVs, in_leaf, in_expected_root, in_path, ".merkle"),
	m_hash_pksig(in_pb, in_params, "pksig_hash", m_pksig, FMT(this->annotation_prefix, ".pksig_hash")),
	m_hash_tag(in_pb, in_params, "tag_hash", flatten({in_msk, m_hash_pksig.result()}), FMT(this->annotation_prefix, ".tag_hash"))
{
}

void anofel_training::generate_r1cs_constraints()
{
	//m_mpk_x_bits.generate_r1cs_constraints();
	//m_mpk_y_bits.generate_r1cs_constraints();
	//m_hash_commit.generate_r1cs_constraints();
	//m_pubkey.generate_r1cs_constraints();
	//m_eddsa.generate_r1cs_constraints();
	//m_mimc.generate_r1cs_constraints();

	m_merkle.generate_r1cs_constraints();
	m_pksig_x_bits.generate_r1cs_constraints();
	m_pksig_y_bits.generate_r1cs_constraints();
	m_hash_pksig.generate_r1cs_constraints();
	m_hash_tag.generate_r1cs_constraints();


	//this->pb.add_r1cs_constraint(
	//		ConstraintT(m_pubkey.result_x(), FieldT::one(), m_mpk.x),
	//		FMT(this->annotation_prefix, "mpk_x = msk.B_x"));
	//this->pb.add_r1cs_constraint(
	//		ConstraintT(m_pubkey.result_y(), FieldT::one(), m_mpk.y),
	//		FMT(this->annotation_prefix, "mpk_y = msk.B_y"));

	//this->pb.add_r1cs_constraint(
	//		ConstraintT(m_hash_commit.result_x(), FieldT::one(), m_comm.x),
	//		FMT(this->annotation_prefix, "in_comm_x = comm_x"));
	//this->pb.add_r1cs_constraint(
	//		ConstraintT(m_hash_commit.result_y(), FieldT::one(), m_comm.y),
	//		FMT(this->annotation_prefix, "in_comm_y = comm_y"));
	//
	//this->pb.add_r1cs_constraint(
	//		ConstraintT(m_mimc.result(), FieldT::one(), m_mimc_A),
	//		FMT(this->annotation_prefix, "mimc(A) = leaf"));

	this->pb.add_r1cs_constraint(
			ConstraintT(m_hash_tag.result_x(), FieldT::one(), m_tag.x),
			FMT(this->annotation_prefix, "in_tag_x = tag_x"));
	this->pb.add_r1cs_constraint(
			ConstraintT(m_hash_tag.result_y(), FieldT::one(), m_tag.y),
			FMT(this->annotation_prefix, "in_tag_y = tag_y"));
	

}


void anofel_training::generate_r1cs_witness()
{
	//m_mpk_x_bits.generate_r1cs_witness();
	//m_mpk_y_bits.generate_r1cs_witness();
	//m_hash_commit.generate_r1cs_witness();
	//m_pubkey.generate_r1cs_witness();
	//m_eddsa.generate_r1cs_witness();
	//m_mimc.generate_r1cs_witness();

	m_merkle.generate_r1cs_witness();
	m_pksig_x_bits.generate_r1cs_witness();
	m_pksig_y_bits.generate_r1cs_witness();
	m_hash_pksig.generate_r1cs_witness();
	m_hash_tag.generate_r1cs_witness();
	
}


//const VariableT& anofel_training::comm_x()
//{
//	return m_hash_commit.result_x();
//}
//const VariableT& anofel_training::comm_y()
//{
//	return m_hash_commit.result_y();
//}

const VariableT& anofel_training::tag_x()
{
	return m_hash_tag.result_x();
}
const VariableT& anofel_training::tag_y()
{
	return m_hash_tag.result_y();
}


