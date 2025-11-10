# Replay Attacks/Signature Malleability

## Overview

Cryptography is central to the functionality of all smart contracts. The cryptographic primitives implemented in protocols are often based on the very same primitives used by the chain to operate in a permissionless way. However, they can sometimes be incorrectly utilized, leading to actions being performed more times than intended, resulting in financial loss or an incorrect contract state.

## Description

Replay attacks occur when an attacker replays a valid transaction or message to deceive the smart contract into performing an action more than once. EVM-based chains have access to a primitive ecrecover, which allows a smart contract to verify some data was verified and signed by the recovered address. This native function, however, does not implement any kind of replay protection.

Typically, replay protection is implemented by introducing a nonce (number used once), which is incremented when a signature is used, preventing the original signature from being used again once the nonce is updated. Signature malleability refers to the ability to modify the signature without invalidating it, allowing a signature to be used twice. This can be introduced when encoding data or casting between types, where some part, or bits, of a value are ignored when checking the signature, but used in their entirety to prevent replay attacks.

## Prevention

To prevent replay attacks and signature malleability vulnerabilities, developers should implement nonce-based transaction management. Nonces can ensure that each transaction is unique and prevent replay attacks. Additionally, implementing proper signature verification checks, such as validating the integrity and authenticity of signatures, can help prevent signature malleability attacks. Contract design should also include mechanisms that prevent unintended actions from being repeated, such as one-time-use tokens or unique transaction identifiers.

## Examples

The highest paid bounty in history at the time, Polygon’s Double-Spend vulnerability, dealt with a bug in Polygon’s WithdrawManager, which verified the inclusion and uniqueness of a burn transaction in previous blocks. The method branchMask that was encoded allowed multiple unique branch masks to encode to the same exit id. This signature malleability would have allowed an attacker to deposit only $100k and reuse signatures to result in a loss of $22.3M.
