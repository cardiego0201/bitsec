# Oracle/Price Manipulation

## Overview

Smart contracts often rely on external data sources, called oracles, to make informed decisions. If these oracles are compromised or manipulated, it may lead to incorrect pricing for swaps, improper reward calculation, the ability to borrow more assets than a collateralization ratio allows, or other general issues with financial transactions. Manipulation of these external data sources is one of the leading causes of DeFi exploits that occur on-chain. Selecting trusted oracles and implementing secure data verification mechanisms are crucial to mitigating the risks associated with oracle/price manipulation.

## Description

Since many protocols are designed to update pricing of assets based on user actions, this can be an obvious but easily overlooked vulnerability, as prices are expected to update based on user interaction. However, when protocols are reliant on those pricing mechanisms, internally or externally, careful consideration should be taken to ensure spot prices cannot be abused. Whether or not price manipulation can be effectively executed can also be highly dependent on current on-chain conditions. Pools with shallow liquidity are at higher risk for manipulation than those which have deeper liquidity. Careful selection of trusted oracles and implementation of secure data verification mechanisms are crucial. Staleness checks, average pricing mechanisms, and read-only reentrancy protections may be important features to implement for robust integration of external pricing mechanisms. Diversification of data sources can also prevent exploits in a single protocol from wreaking havoc on the entire blockchain ecosystem.

## Prevention

To prevent oracle/price manipulation vulnerabilities, developers should carefully select trusted oracles with reputable track records. Implementing secure data verification mechanisms, such as cryptographic proofs or multiple data source aggregation, can help ensure the accuracy and integrity of the received data. Regularly auditing and monitoring oracle contracts and their interactions with smart contracts can also help identify potential vulnerabilities. Assumptions should not be made about the accuracy of data returned from external oracles, and protections should be put in place to prevent temporary price manipulations or stale data from affecting the operation of the protocol.

## Examples

BonqDAO experienced a price oracle manipulation attack which allowed an attacker to momentarily inflate the price of the WALBT token, in order to borrow much more stablecoin (BEUR) than entitled to. The attacker then manipulated the price again, reducing it to a very small value in order to liquidate over 30 under-collateralized troves. What made BonqDAO vulnerable was not the permissionless nature of its price reporting mechanism, but the fact that it considered the protocol contract’s spot price as the last reported value. Because of that, anyone could momentarily inflate or deflate the value of a given price feed.
