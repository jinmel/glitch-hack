// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import './Verifier.sol';

contract AntiScam {
  Verifier private verifier = new Verifier();

  function safeSend(address payable _to,
                    uint256[] memory pubInputs,
                    bytes memory proof) public payable {
    verifier.verify(pubInputs, proof);
  }
}
