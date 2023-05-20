#!/bin/sh

set -ex


INPUT=$1
NETWORK=$2
NAME=$3

ezkl prove --transcript=evm -D input.json -M network.onnx --proof-path $NAME.pf --pk-path pk.key --params-path=kzg.params --circuit-params-path=circuit.params

ezkl create-evm-verifier --deployment-code-path $NAME.code --params-path=kzg.params --vk-path vk.key --sol-code-path $NAME.sol --circuit-params-path=circuit.params
ezkl verify-evm --proof-path $NAME.pf --deployment-code-path $NAME.code


# ezkl gen-srs --logrows 20 --params-path=kzg.params
ezkl setup -D $INPUT -M $NETWORK --params-path=kzg.params --vk-path=vk.key --pk-path=pk.key --circuit-params-path=circuit.params

# Single proof -> single proof we are going to feed into aggregation circuit. (Mock)-verifies + verifies natively as sanity check
ezkl prove --transcript=poseidon --strategy=accum -D $INPUT -M $NETWORK --proof-path $NAME.pf --params-path=kzg.params  --pk-path=pk.key --circuit-params-path=circuit.params

# Aggregate -> generates aggregate proof and also (mock)-verifies + verifies natively as sanity check
ezkl aggregate --logrows=17 --aggregation-snarks=$NAME.pf --aggregation-vk-paths $NAME.vk --vk-path vk.key --proof-path aggr_$NAME.pf --params-path=kzg.params --circuit-params-paths=circuit.params

# Generate verifier code -> create the EVM verifier code
ezkl create-evm-verifier-aggr --deployment-code-path aggr_$NAME.code --params-path=kzg.params --vk-path vk.key

ezkl deploy-verifier-evm -S ./mymnemonic.txt -U myethnode.xyz --deployment-code-path aggr_$NAME.code --sol-code-path aggr_$NAME.sol
