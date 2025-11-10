# Governance Attacks

## Overview

Governance attacks refer to the manipulation or exploitation of the governance mechanisms within a decentralized protocol or smart contract system. These attacks aim to compromise the decision-making processes, voting systems, or parameter adjustments of the governance system, allowing the attacker to gain undue control or benefit from the protocol.

## Description

Governance attacks can take various forms, including allowing proposals to be executed without quorum, allowing execution of proposals without any voting step, or directly manipulating the votes of other participants. These attacks can compromise the decentralized nature of the protocol and lead to centralization of power, or result in financial benefits for the attackers. Governance attacks are particularly relevant in Decentralized Autonomous Organizations (DAOs), where decision-making authority is distributed among token holders.

## Prevention

To prevent governance attacks, it is important to establish a robust, well-defined, and transparent governance framework that outlines the decision-making processes, voting mechanisms, and rules for participation. Implement secure and tamper-resistant voting systems that ensure the integrity of votes. This may involve using cryptographic techniques, zero-knowledge proofs, or multi-signature schemes to enhance security. Ensure a fair and decentralized distribution of tokens to avoid the concentration of voting power in the hands of a few entities. Consider mechanisms such as token vesting or lock-up periods to discourage short-term manipulative behavior. Ensure a majority of governance tokens are not left distributed amongst exchanges such that a malicious actor could acquire enough tokens to unanimously pass proposals.

## Examples

One notable example of a governance attack is the attack on Beanstalk, a permissionless FIAT stablecoin protocol. The hacker submitted two Beanstalk Improvement Proposals: BIP18 and BIP19. The first one proposed the full transfer of funds to the attacker, and the second one was a proposal to send $250k worth of $BEAN tokens to Ukraine’s official crypto donation address. The attacker flash loaned more than $1 billion from Aave, Uniswap, and SushiSwap to gain enough voting power (at least a ⅔ majority) to trigger an emergency governance execution.
