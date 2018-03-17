---
layout: post
title: "B-Money"
categories: []
tags: [concept, references]
description: A scheme for a group of untraceable digital pseudonyms to pay each other with money and to enforce contracts amongst themselves without outside help.
cover: "/assets/images/decentralized.jpg"
cover_source: "https://www.dailydot.com/wp-content/uploads/58a/11/ea87086c485460123cd2978fd7e46dbe.jpg"
comments: true
---

{% include collection-cryptocurrency.md %}

### Introduction

> B-Money, an early proposal by Wei Dai for an anonymous, distributed electronic cash system was referred by Satoshi Nakamoto for creating Bitcoin, the current cryptocurrency giant.

In the referred post, the author identified existing operational problems in an online community named Crypto-Anarchy. Crypto-Anarchy is explained as an online community where participants are identified with psuedonyms which are in no way associated with their true names or physical locations. Because of this seperation of true identity from community identity, the possibility of voilence is rendered impotent. Together these make the role of government permanently forbidden and unnecessary in such a community.

> A community is defined by cooperation of its participants, and an efficient cooperation requires a medium of exchange (money) and a way to enforce contracts.

While traditionally these services were provided by the government or government sponsored institutions, it was not clear theoretically how to implement a similar system among decentralized nodes without using a trusted central authority.

Author proposed two different protocols which can solve these issues, first one is very similar to the current proof-of-work protocol and second is similar to proof-of-stake protocol.

Author poses that the implementation of the first system is impractical, because it makes heavy use of synchronous and unjammable anonymous broadcast channel. 

> Existence of untraceable network is assumed, where senders and receivers are identified only by their digital psuedonyms (i.e. public keys) and every message is signed by its sender and encrypted to its receiver.

### First Protocol

* Analogous to **proof-of-work**.
* A seperate database is maintained by every participant with details of how much money belongs to which **pseudonym**.
* The individual databases collectively define the ownership of the money. The accounts are update subject to the rules in this protocol listed below.
* **The creation of money:** Money is created by broadcasting the solution to a previously unsolved computational problem. Such solutions must be easy to determine how much computing effort it took to solve the problem and must not otherwise have any practical or intellectual value (similar to nonce in a proof of work). Upon broadcasting the solution and verifying it, everyone credits the amount to solver's psuedonym, equivalent to the amount of units it would take to buy the electricity utilized by the most economical computer to solve the problem.
* **The transfer of money:** If owner of psuedonym pk_A (public key of A) sends X units of money to owner of psuedonym pk_B (public key of B), they have to broadcast a message signed with their public key. On receiving this broadcast, all the participants would debit amount X from psuedonym pk_A and credit it to psuedonym pk_B, unless this creates a negative balance in pk_A's account in which case the broadcasted transaction is ignored.
* **The effecting of contracts:** A valid contract binds a **maximum reperation** for each participating party in case of default. It also specifies a third party who will do the arbitration in case any dispute arises. In order for the contract to be effective, all the participating parties and the arbitrator must broadcast their public keys (i.e. psuedonyms). Upon the broadcast of contract and all the associated signatures, every participant debits the account of each party by the amount of his maximum reperation and credits a special account identified by secure has of the contract by the sum of the maximum reperations. The contract is a success if the debits succeed for every party without any negative balances, failing which the contract is ignored and the accounts are rolled back.
* **The conclusion of contracts:** If the contract concludes without any dispute, each party must broadcast a message stating the same, following which each participant credits the accounts of each party by the amount of their maximum reperation, removes the contract account, then credits or debits the account of each party according to the reparation schedule if there is one.
* **The enforcement of contract:** In case of dispute, which cannot be sorted by the arbitrator, each party broadcasts a suggested reparation/ fine schedule and arguments or evidence in his favor. Each participant makes a determination as to the actual reparations and/or fines, and modifies his accounts accordingly.

### Second Protocol

* Similar to **proof-of-stake**.
* The accounts of every psuedonym is maintained by a **subset of the participants (called servers)** instead of everyone as in the previous protocol. Servers are connected (using Usenet-style broadcast channel).
* Format of broadcasted transaction is same as first protocol, but **affected participant of each transaction should verify the changes** on a randomly selected subset of the servers.
* In order to bring a degree of trust in the servers, **each server is required to deposit a certain amount of money in a special account** to be **used as potential fines or rewards** for proof of misconduct.
* Each server must **periodically  publish and commit** to its current money creation and money ownership databases. Also they should verify that his own account balances are correct and that **total sum of account balances in not greater than the total amount of money created**. This would prevent the servers, even in total collusion, from permanently and costlessly expanding the money supply.
* New servers synchronize with the existing servers to used the published databases.


### Alternative B-Money Creation

One of the major problems faced in decentralized money network protocols is reaching a consensus for the cost of a computing effort. The rapid advances of computing technology, often in a private development makes it difficult to gather accurate information about these metrics while making sure they are not outdated.

Author proposes a subprotocol, in which the account keepers (i.e. the participants in first protocol or servers in second protocol) decide and agree on the amount of b-money to be create each period, where the **cost of b-money is determined by an auction**.

Each money creation period is divided up into **four phases**:

* **Planning:** Account keepers compute and negotiate to determine an optimal increase in money supply for a given period. Whether or not  the participants reach a concensus, they broadcast their money creation quota and any macroeconomic calculations done to support the figures.

* **Bidding:** Anyone who wants to create the b-money makes a bid of the form <X, Y>, where X is the amount of b-money he wants to create and Y is an unsolved problem from a predetermined problem class (proof-of-work solution in case of bitcoin), where each problem has a nominal cost in MIPS-year publically agreed on.

* **Computation:** After bidding, the ones who placed the bids in bidding phase solve the problem in their bid and broadcast the solutions.

* **Money Creation:** Each participant accepts the highest bids (among all those who broadcast solutions) in terms of nominal cost per unit of b-money created and credits the bidder's account accordingly.

## REFERENCES:

<small>[B-Money by Wei Dai](http://www.weidai.com/bmoney.txt){:target="_blank"}</small>