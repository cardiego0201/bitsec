# Rounding Error

## Overview

Improper handling of floating-point arithmetic and rounding errors can lead to financial discrepancies or exploitation of contract logic. Precise handling of numerical operations, using fixed-point arithmetic where applicable, is crucial to avoiding such vulnerabilities. Typically, these vulnerabilities may arise in permissionless exchange protocols, where non-standard decimal values can introduce unforeseen consequences.

## Description

Rounding errors occur when smart contracts perform calculations involving floating-point arithmetic and fail to account for precision or rounding. These errors can lead to financial discrepancies, loss of funds, or incorrect rewards calculated within the contract. Smart contracts should use fixed-point arithmetic or alternative mechanisms to handle decimal calculations accurately, ensuring that rounding errors are minimized or eliminated.

## Prevention

To prevent rounding errors, developers should employ fixed-point arithmetic or libraries that provide precise numerical operations. Fixed-point arithmetic uses integer values to represent decimals, avoiding the imprecision associated with floating-point arithmetic. Additionally, thorough testing and validation of numerical operations, including boundary conditions and corner cases, can help identify and address potential rounding errors.

## Examples

Notably, DFX Finance patched a vulnerability which consisted of a rounding error with the EURS token due to the non-standard decimal value of two. Assimilators are necessary to DFX Finance’s Automated Market Maker, as all assets are paired with USDC. The AssimilatorV2 contract is responsible for converting all amounts to a numeraire, or a base value used for computations across the protocol. The issue arises when an integer division leads to zero tokens being transferred from the user, despite the user still receiving curve tokens which represent their portion of the curve pool.
