#!/bin/sh
set -ex

INPUT=$1
NETWORK=$2

# gen proof
ezkl prove --transcript=evm -D $INPUT -M $NETWORK --proof-path 1l_relu.pf --pk-path pk.key --params-path=kzg.params --circuit-params-path=circuit.params

# gen evm verifier
ezkl create-evm-verifier --deployment-code-path 1l_relu.code --params-path=kzg.params --vk-path vk.key --sol-code-path 1l_relu.sol --circuit-params-path=circuit.params

# Verify (EVM)
ezkl verify-evm --proof-path 1l_relu.pf --deployment-code-path 1l_relu.code
