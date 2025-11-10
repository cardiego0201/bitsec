# Frontrunning

## Overview

Frontrunning is a technique where an attacker exploits the time delay between when a pending transaction is observed, and its inclusion in a block. By placing their own transaction with a higher gas price ahead of the victim’s transaction, the attacker can manipulate the outcome of certain contract interactions for their benefit. Frontrunning is a concern for decentralized exchanges, auctions, or any scenario where transaction order matters.

## Description

Frontrunning occurs when an attacker gains an advantage by executing transactions ahead of others, particularly when they have knowledge of pending transactions that are about to be included in a block. In the context of smart contracts, frontrunning can be detrimental in scenarios where transaction order impacts the outcome. For example, in a decentralized exchange, an attacker can observe a victim’s pending transaction intended to buy a specific token at a certain price. They can then quickly submit their own transaction with a higher gas price in order to buy the same token at a lower price before the victim’s transaction executes. This allows the attacker to benefit from the price difference at the expense of the victim. Frontrunning can be executed by anyone observing the mempool, but can also originate from block producers themselves. Some chains have private RPCs which can mitigate risk, but validators and block producers within the private mempools may also have a false assumption of trust. Developers should mitigate potential frontrunning opportunities at the smart contract level and not rely on external mitigations which partially rely on trust or aligned incentives to secure their protocols.

## Prevention

Preventing frontrunning requires a combination of technical and design considerations. Some preventive measures include the use of secret or commit-reveal schemes, implementing schemes where sensitive information such as prices or bids is kept secret until the transaction is confirmed, off-chain order matching, use of flashbots which allow users to bundle transactions together and submit them directly to miners reducing the opportunity for frontrunning, and fee optimization to reduce the likelihood of being outbid by frontrunners.

## Examples

Frontrunning attacks have been observed in various DeFi protocols. RocketPool and Lido were notified of a vulnerability which could have affected both staking services. A malicious node operator could frontrun a deposit with previously prepared deposit data and minimal needed deposit value by using the same validator bls key to steal user deposits.
