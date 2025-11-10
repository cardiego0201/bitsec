// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract BuyMeACoffee is ReentrancyGuard, Ownable {
    event NewCoffee(
        address indexed sender,
        uint256 timestamp,
        string message,
        uint256 amount
    );
    event WithdrawnTips(uint256 amount, uint256 timestamp);
    
    struct Coffee {
        address sender;
        string message;
        uint256 amount;
        uint256 timestamp;
    }
    
    Coffee[] public coffees;
    uint256 public totalCoffee;
    uint256 public constant MIN_TIP = 0.001 ether;
    uint256 public constant MAX_TIP = 1 ether;
    uint256 public constant MAX_MESSAGE_LENGTH = 280;
    
    constructor() Ownable(msg.sender) {}
    
    function buyCoffee(string calldata _message) public payable nonReentrant {
        require(msg.value >= MIN_TIP, "Tip too small");
        require(msg.value <= MAX_TIP, "Tip too large");
        require(bytes(_message).length <= MAX_MESSAGE_LENGTH, "Message too long");
        require(bytes(_message).length > 0, "Message required");
        
        totalCoffee++;
        coffees.push(Coffee(
            msg.sender,
            _message,
            msg.value,
            block.timestamp
        ));
        
        emit NewCoffee(msg.sender, block.timestamp, _message, msg.value);
    }
    
    function withdrawTips() public onlyOwner nonReentrant {
        uint256 balance = address(this).balance;
        require(balance > 0, "No tips to withdraw");
        
        (bool success, ) = owner().call{value: balance}("");
        require(success, "Withdrawal failed");
        
        emit WithdrawnTips(balance, block.timestamp);
    }
    
    function getCoffees() public view returns (Coffee[] memory) {
        return coffees;
    }
    
    receive() external payable {
        revert("Use buyCoffee() to send tips");
    }
}