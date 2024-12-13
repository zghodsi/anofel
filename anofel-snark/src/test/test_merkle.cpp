#include "gadgets/merkle_path_check.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include "stubs.hpp"
#include "utils.hpp"


template<typename HashT>
libff::bit_vector hash256(uint8_t *input_buffer, size_t input_len) {
    libff::bit_vector operand = bytes_to_bv(input_buffer, input_len);
    size_t size = operand.size();
    operand.push_back(1);
    //libff::bit_vector s = libff::int_list_to_bits({size}, 32);
    //for (int i = size + 1; i < HashT::get_block_len() - s.size(); i++) {
    for (int i = size + 1; i < HashT::get_block_len() ; i++) {
        operand.push_back(0);
    }
    //operand.insert(operand.end(), s.begin(), s.end());
    libff::bit_vector res = HashT::get_hash(operand);
    return res;
}

template<typename HashT>
void calcAllLevels(std::vector<std::vector<libff::bit_vector>>& levels, size_t level) {
    //level 1 upper layer
    for (int i = level; i > 0; i--) {
        for (int j = 0; j < levels[i].size(); j += 2) {
            libff::bit_vector input = levels[i][j];
            input.insert(input.end(), levels[i][j+1].begin(), levels[i][j+1].end());
            levels[i-1].push_back(HashT::get_hash(input));
        }
    }
}

int main () {
    ppT::init_public_params();
    typedef sha256_two_to_one_hash_gadget<FieldT> HashT;
    const size_t tree_depth = 3;

    ProtoboardT pb;
    merkle_path_check<HashT> MT(pb, tree_depth);
    MT.generate_r1cs_constraints();

    
    libff::bit_vector leaf, root, address_bits(tree_depth);
    size_t address;
    std::vector<merkle_authentication_node> path(tree_depth);

    //generate witness
    std::vector<std::vector<libff::bit_vector>> levels(tree_depth);
    //level 2 leaves left most --> right most
    int leaf_count = std::pow(2, tree_depth);
    for (int i = 0; i < leaf_count; i++) {
        uint8_t n = 3;
        uint8_t *buffer = new uint8_t[n];
        for( size_t i = 0; i < n; i++ ) {
            buffer[i] = i % 0xFF;
        }

        //libff::bit_vector tmp = hash256<HashT>("a");
        libff::bit_vector tmp = hash256<HashT>(buffer, n);
        levels[tree_depth - 1].push_back(tmp);
    }

    calcAllLevels<HashT>(levels, tree_depth-1);
    libff::bit_vector input = levels[0][0];
    input.insert(input.end(), levels[0][1].begin(), levels[0][1].end());
    root = HashT::get_hash(input);

    address = 1;
    leaf = levels[tree_depth-1][address];
    std::cout << address << std::endl;
    int addr = address;
    for (int i = 0; i < tree_depth; i++) {
        int tmp = (addr & 0x01);
        address_bits[i] = tmp;
        addr = addr / 2;
        std::cout << address_bits[tree_depth-1-i] << std::endl;
    }

    //Fill in the path
    size_t index = address;
    for (int i = tree_depth - 1; i >= 0; i--) {
        path[i] = address_bits[tree_depth-1-i] == 0 ? levels[i][index+1] : levels[i][index-1];
        index = index / 2;
    }

    MT.generate_r1cs_witness(pb, leaf, root, path, address, address_bits);

    bool verified = stub_test_proof_verify(pb);
    std::cout << "Verification status: " << verified << std::endl;

    return 0;
}

