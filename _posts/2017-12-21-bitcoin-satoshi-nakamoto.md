---
layout: post
title: "Bitcoin"
categories: []
tags: [concept, references, papers]
description: A Peer-to-Peer Electronic Cash System that prevents double spending using the hash-based proof-of-work driven network transactions, that are authenticated by global consensus.
cover: "/assets/images/bitcoin.jpg"
cover_source: "https://4.bp.blogspot.com/-3aBTdvTvgqM/VNcb2gxRaTI/AAAAAAABccs/PPbDPAL-6Bo/s0/Bitcoin_UHD.jpg"
comments: true
mathjax: true
---

{% include collection-cryptocurrency.md %}

### Introduction

> A purely peer-to-peer version of electronic cash would allow online payments to be sent directly from one party to another without going through a financial institution.

Part solution to this task is given by digital signatures, but the other half of the solution posed a major question in developing such systems - the part that requires **a system to prevent double spending**. 

In the current centralized money tender distributed, this role of preventing the double spending is done by a centralized third-party authority. But in a peer-to-peer network this centralized server would not be available and thus would require a different methodology.

Bitcoin, proposed by **Satoshi Nakamoto**, was to solve this very problem on the peer-to-peer network. The network timestamps transactions by hashing them into an ongoing chain of hash-based proof-of-work, forming a record that cannot be changed without redoing the proof-of-work.

### Current System

Current economy is based on trusted third-parties to process electronic payments. These systems suffer from the inherent weaknesses of the trust based model. Completely non-reversible transactions are not possible because banks cannot avoid mediating disputes.

Because of cost of mediation, the transaction costs are high which inturn limit the minimum practical transaction size thus cutting off the possibility of small casual transaction.

> With the possibility of reversal, the need for trust spreads.

### Proposed System

Bitcoin proposed a system where **trust is replaced by cryptographic proof**, allowing any two willing parties to transact directly without the need of a supporting trusted third party.

> Computationally irreversible transactions prevent frauds.

Bitcoin proposed to tackle the issue of double-spending, using a peer-to-peer distributed timestamp server to generate computational proof of the chronological order of transactions.

The system is secure as long as honest nodes collectively control more CPU power than any cooperating group of attacker nodes.

### Transactions

> Electronic coin is defined as a chain of digital signatures.

A coin is transferred by an owner signing a hash of previous transaction and the public key of the next owner and adding these to then end of the coin.

The signature can be verified by the payee to check the chain of ownership.

The payee can still however not check for double-spending. 

Traditional solution of this involves introducing a trusted central authority, or mint, that checks every transaction for double-spending. In such a system, after every transaction the coin must be returned to the mint to issue a new coin, and only coins issued directly from the mint are trusted not to be double-spent. This still does not solve the basic problem of dependency on company running the mint, which is basically serving the same purpose as that of a bank in the current system.

![Flow of Transactions](/assets/2017-12-21-bitcoin-satoshi-nakamoto/fig-1-transaction.png)

The only way to confirm the absence of a transaction is to have a copy of all the transactions. To accomplish this in a decentralized system, the transactions must be publically announced and there is a need for a system for participants to agree on a single history of the order in which they were received. So, at the time of the transaction, the payee needs a proof that the majority of the nodes agreed it was the first received one.

### Timestamp Server

A timestamp server basically takes a hash of a block of items to be timestamped and widely publishes it, where the timestamp proves that the data must have existed at the time, in order to get the hash.

![Timestamp Server](/assets/2017-12-21-bitcoin-satoshi-nakamoto/fig-2-blockchain.png)

> Each timestamp includes previous timestamp in its hash, forming a chain (blockchain), with each timestamp reinforcing the ones before it.

### Proof-of-Work

> A system to deter denial of service attacks and other service abuses such as spam on a network by requiring some work from the service requester, usually meaning processing time by a computer.

In case of bitcoins, the proof-of-work involves scanning for a value that when hashed along with the contents of the transaction blocks using cryptographic hashes such as SHA-256, gives a hash that begins with a specific number of zero bits. **The average work required is exponential in the number of zero bits.** Proof-of-Work is generally a problem that is fairly tough to solve but very easy to verify.

```python
import time
import hashlib
import json
import matplotlib.pyplot as plt

transaction = [
    'A paid B 25 A_public_key', 
    'A paid C 50 A_public_key', 
    'C paid B 10 C_public_key'
]
transaction = json.dumps(transaction).encode()

difficulty_bits = list(range(0, 8))
time_taken = []
for num_bits in difficulty_bits:
    proved = False
    proof = 0
    start_time = time.time()
    while not proved:
        string = transaction + str(proof).encode()
        current_hash = hashlib.sha256(string).hexdigest()
        if current_hash.startswith('0'*num_bits):
            print(current_hash)
            print(proof)
            proved = True
            time_taken.append(time.time()-start_time)
        proof = proof+1

plt.plot(difficulty_bits, time_taken)
plt.show()

"""
Outputs
-------
cb87ca935b8b44e94f5c670ecd375ba88ab7616aeab4c4c4369fe79e9c153f95
0
0c1d9b1549d14612571040e8352af4175a41b84a805a08880d3acdd5aad998be
5
00a51beea09d1fae73f3e18682445a3eb9bffc2cabf17f87f90f193c9eca1af3
396
00010ff4f55c002484ee13841be9a3c46aba3038a1046d604ac8cee7349a2228
514
0000a3c456eff341c9988b194231b02eb547a36a8134d095a5e1c4efb05dc7e3
22344
000002128ccfc4d1e662a6ceb8053860b887929b691c2f8506b0843e9a57fe34
3355879
000000bfc98081b9aff064455708d2d8fcc2afeac81b5a0b03a6fd926db1ce36
19659271
0000000cbd6302c37b40102ef55040e4315b63b39640110b679e033c44a8b2b0
310723836

Time Taken
----------
[0.0003619194030761719,
 0.00020599365234375,
 0.0012040138244628906,
 0.0016019344329833984,
 0.07679605484008789,
 8.218470811843872,
 47.986721992492676,
 740.9898428916931]
"""
```

![Proof-of-work plot](/assets/2017-12-21-bitcoin-satoshi-nakamoto/fig-8-proof-of-work-plot.png)

Basic needs of such system of proof-of-work are fulfilled by cryptographic hashes, that are often seen as one way functions, i.e. the only way to get the input given an output is to use the bruteforce approach and check all possible values of inputs and arriving at the correct candidate.

> A nonce is an arbitrary number that can only be used once

In the bitcoin's timestamp network, the proof-of-work is implemented by incrementing a nonce in the block until a value is found that gives the block the required number of leading zero bits.

![Proof-of-work Flow](/assets/2017-12-21-bitcoin-satoshi-nakamoto/fig-3-proof-of-work.png)

> Once the CPU effort has been expended to satisfy the proof-of-work, the block cannot be changed without redoing the work.

As the later blocks consume the hashes of previous blocks, changing a transaction in earlier block would require redoing the work for all the leading blocks.

### Majority Decision Making

Proof-of-Work also gives a solution to determining representation in majority decision making. The majority decision is represented by the longest chain, i.e. the greatest proof-of-work. So if the majority CPU power is held by honest nodes, then the honest chain would grow faster and outpace any chain being proposed by a pool of attacking nodes.

> To compensate hardware power and varying number of nodes over time, the proof-of-work difficulty is determined by a moving average targeting an average number of blocks per hour, i.e. if the blocks are generated too fast, the difficulty increases for subsequent blocks.

### Network Protocols

Steps to run the network are as follows:

* New transactions are broadcast to all nodes.
* Each node collects new transaction into a block.
* Each node works on finding a difficult proof-of-work for its block.
* Upon finding the solution, it broadcasts the block to all nodes.
* Nodes accept a new block if all the transactions on it are valid and not already spent.
* Nodes express their acceptance of the block by working on creating next block in the chain, using the hash of the accepted block as the previous hash.

> Nodes always consider the longest chain to be the correct one and keep on working to extend it.

If a node does not receive a block, it will request it when it receives the next block and realizes that it missed one.

### Mining Incentive

By convention of the protocol, the first transaction of a block is a special transaction that starts a new coin owned by the creator of the block. 

It is this, that acts as an incentive for the nodes to support the network, and provides a way to initially distribute coins into circulation.

Incentive can also be funded in the form of a transaction fees. 

Incentive will also encourage nodes to stay honest. Because an attacker with more compute power would find it more favourable to play by rules and make more transaction fee, instead of competing with rest to fraud the system and invalidating his own wealth in the process by sabotaging the system.

### Memory Optimizations

After the latest transaction is reinforced with several blocks stacking on top of it, the spent transactions can be discarded from the disk. 

> Transactions are hashed in a Merkle Tree, with only root included in the block's hash.

![Memory Optimizations](/assets/2017-12-21-bitcoin-satoshi-nakamoto/fig-4-merkle-tree.png)

A block header with no transaction is about 80 bytes, and blocks are generated every 10 minutes, then every hour about 6 blocks would be generated. So the total space required by blocks generated in a year would be about 4.2MB (i.e 80 bytes * 6 * 24 * 365). With the current capabilities of system memories, storage would not pose a problem to the system.

### Payment Verification

The system makes it possible to verify payments without running the full network node. 

Verification can be done by keeping a copy of the block headers of the longest proof-of-work chain, and obtaining the Merkle branch linking the transaction to the block it's timestamped in.

![Payment Verification Process](/assets/2017-12-21-bitcoin-satoshi-nakamoto/fig-5-validating-transaction.png)

By linking the transaction to a place in the chain, one can see if network nodes have accepted it by checking the proof-of-work, which would be further confirmed by blocks built on top of it in the chain.

Again, the verification is only reliable as long as honest nodes control the network. Because the simplified verification method can be fooled by an attacker's fabricated transactions for as long as the attacker can overpower the compute resources of the unified honest nodes.

### Combining and Splitting Value

Even though it is possible to handle each coin individually, it would be unweildy to make a seperate transaction for every cent in a transfer.

> To allow value to be split and combined, transactions contain multiple inputs and outputs. 

![Value Handling Protocols](/assets/2017-12-21-bitcoin-satoshi-nakamoto/fig-6-splitting-and-joining.png)

Usually, there is a single input from a larger previous transaction or multiple inputs combining smaller amounts, and at most two outputs: **one for the payment, and one to return the change, if any, back to the sender**.

### Privacy

Traditional banking models achieve privacy through limiting access to information to the parties involved and the trusted third party. 

![Privacy Models](/assets/2017-12-21-bitcoin-satoshi-nakamoto/fig-7-privacy.png)

In the proposed system, it is achieved by anonymizing the public keys that are used to sign a transaction. So, the public ledger can see someone is sending an amount to someone else, but without information linking the transaction to anyone (similar to tape in a stock exchange).

Some linking is still unavoidable with multi-input transactions, which reveals that the inputs originated from the same owner. So if one owner's key is revealed, linking will allow to discover other transactions associated with that owner.

### Mathematics of Bitcoin

If there were a scenario where an attacker is generating a alternate chain faster than the honest chain, it could still not generate value out of thin air or take someone else's money because it will be rejected by the corresponding node which owns that token. 

So all it is capable of doing is to double-spend its own money.

Basically the race between the honest chain and the alternate chain can be characterized as a **Binomial Random Walk**, where success event is honest chain being extended by a block and failure event is the alternate chain extented by a block gaining a lead on the honest chain.

In such a case, the probability of an attacker catching up with the honest chain by making up for the deficit can be modeled similar to a **Gambler's Ruin Problem**.

The **probability that the attacker ever reaches a breakeven** is given by, 

$$
q_z = 
\begin{cases}
1 & \text{ if } p \leq q \\
(q/p)^z & \text{ if } p \gt q
\end{cases}
$$

where, 

* \\(p\\) is the probability of an honest chain extending
* \\(q\\) is the probability of a attacker chain extending
* \\(q_z\\) is the probability of the attacker catching up with the honest chain from \\(z\\) blocks deficit.

Given the assumption that the pool of honest compute power is greater than that of attackers, it is safe to assume that \\(p \gt q\\). So the probability drops exponentially as the number of blocks of deficit for an attacker increases. So after a fair lead, the probability of attacker to catch up is vanishingly small.

### Time for Establishing Trust

Assume that the sender of a transaction is an attacker who wants to make the receiver believe that he paid him for a while, and switch it back to himself after sometime.

So initially the receiver will generate a new key pair and give the public key to the attacker for making the transfer. This prevents the attacker from preparing a chain of blocks in advance because of the dynamic nature of the public key generated.

So once the transaction is made, the attacker starts working on a alternate chain where the transactions are different.

Meanwhile the receiver waits for the transaction to be added to a block and another \\(z\\) blocks to be chained on top of that block. 

The receiver does not know the progress of the attacker, but assuming the honest nodes took the average time to build a block, the attacker's progress will be a Poisson Distribution with the expected value,

$$ \lambda = z {q \over p}$$

Probability that the attacker could catch up can be calculated by mutliplying the Poisson density for each amount of progress by the probability that he could catch up from that point, given by,

$$
\sum_{k=0}^\infty \frac{\lambda^k\,e^{-\lambda}}{k!} \cdot \begin{cases} (q/p)^{(z-k)} & \text{ if } k \leq z \\ 1 & \text{ if } k \gt z \end{cases}
$$

Rearranging to avoid infinite summation of the distribution,

$$ 1 - \sum_{k=0}^z \frac{\lambda^k\,e^{-\lambda}}{k!} (1 - (q/p)^{(z-k)})$$

```python
import math

q = 0.1
p = 1-q

def factorial(k):
    fac = 1
    for _ in range(1, k+1):
        fac *= _
    return fac

def calculate(z):
    lamda = z * q/p
    sum = 1
    for k in range(z+1):
        sum -= math.pow(lamda, k) * math.exp(- lamda) / factorial(k) * (1 - math.pow((q/p), z-k))
    return sum

for _ in range(0, 30, 5):
    print(_, "\t=> ", round(calculate(_),8))

"""
Output
------
0   =>  1.0
5   =>  0.17735231
10  =>  0.04166048
15  =>  0.01010076
20  =>  0.0024804
25  =>  0.00061323
"""
```

**Calculating the values, it is observed that the probability drops off exponentially with z.**

### Conclusions

* Bitcoin does not need a third party to enforce trust.
* Coin framework is based on digital signatures.
* A peer-to-peer network using proof-of-work records a history of transaction.
* Nodes can leave or rejoin the system according to their wish. 
* Any needed rules and incentives can be enforced with this consensus mechanism.

## REFERENCES:

<small>[Bitcoin: A Peer-to-Peer Electronic Cash System](https://bitcoin.org/bitcoin.pdf){:target="_blank"}</small>