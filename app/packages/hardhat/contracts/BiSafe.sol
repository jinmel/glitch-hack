pragma solidity >=0.8.0 <0.9.0;
//SPDX-License-Identifier: MIT

import "hardhat/console.sol";
import "./Verifier.sol";
// import "@openzeppelin/contracts/access/Ownable.sol";
// https://github.com/OpenZeppelin/openzeppelin-contracts/blob/master/contracts/access/Ownable.sol

contract BiSafe {
  Verifier private verifier = new Verifier();

  function safeTransfer(address payable _to,
                        uint256[] memory pubInputs,
                        bytes memory proof) public payable {
    verifier.verify(pubInputs, proof);
    // You can write your transfer function here.
  }
}
