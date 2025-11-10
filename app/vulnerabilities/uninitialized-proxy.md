# Uninitialized Proxy

## Overview

Using proxy contracts without proper initialization can introduce vulnerabilities, as uninitialized storage variables can be manipulated by attackers. Implementing secure proxy patterns and conducting comprehensive initialization checks are vital to preventing unauthorized access to uninitialized contract states. Uninitialized proxy bugs have led to the highest bounty payout so far of $10m.

## Description

Uninitialized proxy contracts refer to instances where the state variables of a proxy contract are not properly initialized before use. This can create security vulnerabilities, as uninitialized storage variables may contain sensitive data or control critical contract behavior. Attackers can exploit these vulnerabilities by manipulating uninitialized storage variables to gain unauthorized access or execute unintended actions. To mitigate this risk, it is crucial to ensure that proxy contracts are properly initialized before they are used in production environments.

## Prevention

To prevent uninitialized proxy vulnerabilities, developers should ensure that all storage variables within the proxy contract are correctly initialized before deploying and using the proxy. This includes initializing sensitive data, access control permissions, and any other critical state variables. Developers should also implement comprehensive initialization checks within the proxy contract to verify that all required variables and dependencies have been properly initialized. They should also implement monitoring tools to provide a secondary level of verification that initialization has occurred properly. This can be achieved through constructor functions or initialization functions that are called after deployment through automated scripts or transaction bundling.

## Examples

The Wormhole Uninitialized Proxy bug report is the highest paid bounty so far in history: $10m paid to the whitehat who submitted the report. When UUPS proxy contracts are deployed, the “constructor” is instead a regular Solidity function that exists in the implementation. The implementation provides the initialize() function which initializes the owner. Wormhole did not initialize the implementation contract of their implementation, which would have allowed an attacker to gain control of the implementation, and self-destruct the contract which would prevent the proxy contracts from being able to execute, as the logic they point to no longer exists.
