# Incorrect Calculation

## Overview

Incorrect calculations are a close second in confirmed bug reports that we see on Immunefi.

Incorrect or inconsistent calculations within a smart contract can result in unintended consequences, such as inaccurate token balances, incorrect reward distribution, or unexpected results during contract execution. Incorrect calculations are typically paired with underexplored code paths and are closely related to improper input validation bugs. However, incorrect calculations deal with vulnerabilities in which it was intended for the user to be able to perform some action, but as a result of the incorrect calculation, the user may receive much more value in return from that action than expected.

## Description

In smart contracts, incorrect calculations occur when mathematical operations are performed incorrectly, leading to unexpected or incorrect results. These vulnerabilities can arise due to various reasons, such as incorrect assumptions about the precision, range of values, or inconsistent calculations across different parts of the contract. Incorrect calculations can also occur when contracts fail to consider edge cases or handle corner cases properly. In some instances, contracts may not account for extreme values, or fail to handle overflows, or underflows, leading to unexpected behavior or security risks. These vulnerabilities can be exploited by attackers to manipulate calculations or gain unauthorized advantages within the contract. Proper mathematical precision and careful consideration of corner cases are essential to prevent such vulnerabilities.

## Prevention

Unit testing along with fuzzing or symbolic execution can help prevent missing edge cases from creeping into the codebase. Additionally, formal verification of mathematical equations would help provide guarantees to states which can exist. Use well-tested and secure mathematical libraries or built-in functions provided by the blockchain platform to perform calculations accurately. These libraries often have built-in protections against common calculation errors, such as overflow or underflow.

## Examples

88MPH Theft Of Unclaimed MPH Rewards Bugfix Review demonstrates a case of incorrect calculation, in which MPH rewards were calculated with the incorrect rewardPerToken and would have allowed an attacker to entirely drain the vesting contract of unclaimed MPH rewards.
