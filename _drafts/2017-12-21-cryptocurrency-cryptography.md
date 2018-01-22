---
layout: post
title: "Cryptography powering the Cryptocurrencies"
categories: []
tags: [concept, references]
description: The trust in the cryptocurrency system is a by-product of the robust cryptographic concepts that make these systems extremely secure, and practically unhackable.
cover: "/assets/images/hash.jpg"
cover_source: "https://www.walldevil.com/wallpapers/w03/953124-cracking-cryptography-encryption-hacking-hexadecimal.jpg"
comments: true
mathjax: false
---

### What is Cryptography?

> Cryptography is associated with the process of converting ordinary plain text into unintelligible text and vice-versa. 

It is a method of storing and transmitting data in a particular form so that only those for whom it is intended can read and process it. Cryptography not only protects data from theft or alteration, but can also be used for user authentication.

Particularly, in computer science, cryptography is closely associated with scrambling plaintext (ordinary text, sometimes referred to as **cleartext**) into **ciphertext** (a process called **encryption**), then back again (known as **decryption**).

Four major objectives of cryptography are:

* **Confidentiality:** Data can be understood by the concerned parties.
* **Integrity:** In the process of encryption and decryption, there can be no alteration of data stored and transmitted.
* **Non-repudiation:** Creator/sender cannot later deny the intent of creating/sending the information.
* **Authentication:** Sender and receiver should be able to confirm each other's identities.

### Encryption

In cryptography, encryption is the process of encoding a message or information in such a way that only authorized parties can access it and those who are not authorized cannot. Encryption does not itself prevent interference, but denies the intelligible content to a would-be interceptor.

Majorly, there are three types of encrytion:

* Symmetric Encryption
* Asymmetric Encryption
* Hashing

### Symmetric Encryption

Also known as the **shared secret encryption**, this form of encryption is uses a secret key, called **shared secret**, to scramble data into unitelligible gibberish. But the receiver has to have access to the same secret key in order to decipher the sequence and gain information from it.

![Symmetric Encryption Flowchart](/assets/2017-12-21-cryptocurrency-cryptography/fig-1-symmetric-encryption.jpg?raw=true)

It is called symmetric encryption because the same key is used on both the ends, i.e. during encryption and decryption. This property presents a challenge to this form of encryption, because the secret key itself can get intercepted if the channel of communication is not secure, and would beat the entire underlying purpose of encryption.

To overcome this very drawback of the symmetric encryption, the pioneers of the field came up with the **Public Key Encryption**, popularly known as the asymmetric cryptography.

### Asymmetric Encryption

Under this type of encryption splits the key into two parts, one is made public and other is kept private. Messages are encrypted with the receivers public key which can be only decrypted by the corresponding private key.

![Asymmetric Encryption Flowchart](/assets/2017-12-21-cryptocurrency-cryptography/fig-2-asymmetric-encryption.jpg?raw=true)

This differs from the symmetric encryption, because one does not need a party's private key to send them a message. Once the public key of a person is broadcasted, all the messages to them is secure because it can be only deciphered by the owner of the complementary private key. So there is no need of prior exchange of confidential data, to set up a secure line of communication.

Public key encryption is usually implemented using the one-way functions. In mathematics, these functions are easy to compute in one direction but very difficult to compute in the inverse direction. This property is what helps publish the public key without comprimising the encryption of the message as long as the private key is securely managed.

>  A common one-way function used today is factoring large prime numbers. It is easy to multiply two prime numbers together and get a product. However, to determine which of the many possibilities are the two factors of the product is one of the great mathematical problems.

**RSA is currently the most widely used public key encryptions algorithms.**

### RSA Cipher



### Understanding Digital Signature

A digital signature in essence tries to replicate the properties of the on-paper signature, i.e. the following properties, 

* **Verification:** Signature should verify the identity of the person.
* **Non-forgeable:** Signature should not be easy to copy and forge by another party.
* **Non-repudiation:** Once signed by a person, it cannot be claimed to be fake or retrieved back.

Thinking on these lines the traditional on-paper signatures, no matter how intricate, are **not that efficient or relaible** owing to the ease with which it can be copied and forged. Also there is no measure to determine authenticity with simple visual tools owing to the variance between two instances of signatures by the same person.

These shortcomings are solved by digital signatures, which cryptography provides in the form of keys.

## REFERENCES:

<small>[Definition of Cryptography](https://economictimes.indiatimes.com/definition/cryptography){:target="_blank"}</small>
<small>[Encryption - Wikipedia](https://en.wikipedia.org/wiki/Encryption){:target="_blank"}</small>
<small>[What is cryptography? - Definition from WhatIs.com](http://searchsoftwarequality.techtarget.com/definition/cryptography){:target="_blank"}</small>
<small>[http://books.gigatux.nl/mirror/securitytools/ddu/ch09lev1sec1.html](http://books.gigatux.nl/mirror/securitytools/ddu/ch09lev1sec1.html){:target="_blank"}</small>
