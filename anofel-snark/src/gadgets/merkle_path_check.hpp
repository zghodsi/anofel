#ifndef MERKLE_HPP_
#define MERKLE_HPP_

#include "types.hpp"

#include <libsnark/gadgetlib1/gadgets/merkle_tree/merkle_tree_check_read_gadget.hpp>
#include <libsnark/gadgetlib1/gadgets/hashes/sha256/sha256_gadget.hpp>

using namespace libsnark;

template<typename HashT>
class merkle_path_check : public GadgetT
{
public:
    const size_t digest_size;
    const size_t tree_depth;
    std::shared_ptr<digest_variable<FieldT>> root_digest;
    std::shared_ptr<digest_variable<FieldT>> leaf_digest;
    pb_variable_array<FieldT> address_bits_var;
    std::shared_ptr<merkle_authentication_path_variable<FieldT, HashT>> path_var;
    std::shared_ptr<merkle_tree_check_read_gadget<FieldT, HashT>> merkle;

    merkle_path_check(ProtoboardT &pb, const size_t& depth):
        GadgetT(pb, FMT("merkle_path_check")),
        digest_size(HashT::get_digest_len()),
        tree_depth(depth)
    {
        root_digest = std::make_shared<digest_variable<FieldT>>(pb, digest_size, "root");
        leaf_digest= std::make_shared<digest_variable<FieldT>>(pb, digest_size, "leaf");
        path_var = std::make_shared<merkle_authentication_path_variable<FieldT, HashT>>(pb, tree_depth, "path");

        address_bits_var.allocate(pb, tree_depth, "address_bits");
        merkle = std::make_shared<merkle_tree_check_read_gadget<FieldT, HashT>>(pb, tree_depth, address_bits_var, *leaf_digest, *root_digest, *path_var, ONE, "merkle");
        pb.set_input_sizes(root_digest->digest_size);
    }

    void generate_r1cs_constraints() {
        path_var->generate_r1cs_constraints();
        merkle->generate_r1cs_constraints();
    }

    void generate_r1cs_witness(ProtoboardT &pb, libff::bit_vector& leaf,
                               libff::bit_vector& root, merkle_authentication_path& path,
                               const size_t address, libff::bit_vector& address_bits) {
        root_digest->generate_r1cs_witness(root);
        leaf_digest->generate_r1cs_witness(leaf);
        address_bits_var.fill_with_bits(pb, address_bits);
        assert(address_bits_var.get_field_element_from_bits(pb).as_ulong() == address);
        path_var->generate_r1cs_witness(address, path);
        merkle->generate_r1cs_witness();
    }
};


// MERKLE_HPP_
#endif
