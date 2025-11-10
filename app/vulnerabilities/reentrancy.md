# Reentrancy

## Overview

Reentrancy has a long history in Ethereum and is the vulnerability class responsible for The DAO Hack, one of the biggest attacks on the early Ethereum network in 2016. Reentrancy allows an attacker to repeatedly call a vulnerable contract before the previous call completes, leading to unexpected state changes and unauthorized fund transfers.

## Description

Reentrancy vulnerabilities occur when a contract allows external calls to be made to it during its execution without properly managing the state changes and flow of execution. Reentrancy allows an attacker to repeatedly call a vulnerable contract before the previous call completes, leading to unexpected state changes and unauthorized fund transfers. Implementing secure state management patterns and applying mutex locks can mitigate the risk of reentrancy attacks. Some tools which can help you identify where reentrancy may exist in your contracts include Slither, Mythril, and Pyrometer. You can read more about reentrancy in this article The Ultimate Guide to Reentrancy.

## Prevention

To prevent reentrancy vulnerabilities, developers should follow secure state management patterns such as the “Checks-Effects-Interactions” (CEI) pattern. This pattern ensures that all state changes are made before any external calls are executed, preventing reentrancy attacks. Additionally, implementing mutex locks or using the “ReentrancyGuard” pattern can further safeguard against reentrancy by blocking reentrant calls.

## Examples

The Omni Protocol experienced a hack that led to the loss of $1.4m dollars, as a result of a reentrancy attack on its Ethereum smart contracts. The vulnerable code used the ERC721’s safeTransferFrom method, which makes a call to smart contracts implementing the onERC721Received interface. This external call hands over execution to the receiver and introduces the reentrancy vulnerability.
