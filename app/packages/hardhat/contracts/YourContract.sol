pragma solidity >=0.8.0 <0.9.0;
//SPDX-License-Identifier: MIT

import "hardhat/console.sol";
import "./Verifier.sol";
// import "@openzeppelin/contracts/access/Ownable.sol";
// https://github.com/OpenZeppelin/openzeppelin-contracts/blob/master/contracts/access/Ownable.sol

contract YourContract {
  Verifier private verifier = new Verifier();

  function safeSend(address payable _to,
                    uint256[] memory pubInputs,
                    bytes memory proof) public payable {
    verifier.verify(pubInputs, proof);
  }


  constructor() payable {
    // what should we do on deploy?
  }

  // to support receiving ETH by default
  receive() external payable {}
  fallback() external payable {}
}
