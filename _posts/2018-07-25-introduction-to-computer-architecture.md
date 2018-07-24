---
layout: post
title: "Introduction to Computer Architecture"
categories: [nptel-computer-architecture]
tags: [computer-science, nptel]
description: Computer architecture is the view of a computer as presented to software designers, while computer organization is the actual hardware implementation of a computer.
cover: "/assets/images/computer-architecture.jpg"
cover_source: "http://hdwallpapersrocks.com/wp-content/uploads/2013/09/3D-computer-chip-HD-wallpaper.jpg"
comments: true
mathjax: true
---

### What is a Computer? 

A computer is a general purpose device that can be programmed process information, yield meaningful results.

The three important take-aways being:

- programmable device
- process information
- yield meaningful results

So the important parts for the working of a computer are:

- Program: a list of instructions given to computer
- Information Store: the data it has to process
- Computer: processes information into meaningful results.

A fully functional computer includes at the very least:

- Processing Unit (CPU)
- Memory
- Hard disk

Other than these some input output (I/O) devices can also be a part of the system, such as:

- Keyboard: Input
- Mouse: Input
- Monitor: Output
- Printer: Output

### Memory vs Hard Disk

- Storage Capacity: more on hard disk, less on memory
- Volatile: data on hard disk is non-volatile, while on memory is volatile
- Speed: speed of access and other operations are slower on hard disk when compared to memory.

### Brain vs Computer

- Brain is capable of doing a lot of abstract work that computers cannot be programmed to do.
- Speed of basic calculations is much higher in a computer which is its primary advantage.
- Computers do not get tired or bored or disinterested.
- Humans can understand complicated instructions in a variety of semantics and languages.

### Program

- Write a instruction in a high level language like C, C++, Java etc. (done by human interface)
- Compile it into an executable (binary) that converts it into byte-code, i.e. the language computers understand. (done by compilers)
- Execute the binary. (done by processor)


### Instruction Set Architecture (ISA)

The semantics of all the instructions supported by a processor is known as instruction set architecture (ISA). This includes the semantics of the instructions themselves along with their operands and interfaces with the peripherals.

> ISA is an interface between software and hardware.

Examples of ISA:

- arithmetic instructions
- logical instructions
- data transfer/movement instructions

Features of ISA:

- Complete: it should be able to execute the programs a user wants to write
- Concise: smaller set of instructions, currently they fall in the range 32-1000
- Generic: instructions should not be too specialized for a given user or a given system.
- Simple: instructions should not be complicated

There are two different paradigms of designing an ISA:

- RISC: Reduced Instruction Set Computer has fewer set of simple and regular instructions in the range 64 to 128. eg. ARM, IBM PowerPC. Found in mobiles and tablets etc.
- CISC: Complex Instruction Set Computer implements complex instructions which are highly irregular, take multiple operands. Also the number of instructions are large, typically 500+. eg. Intel x86, VAX. Used in desktops and bigger computers.


### Completeness of ISA

**How do we ensure the completeness of an ISA?** Say, there are two instructions addition and subtraction, while it is possible to implement addition using substraction (a + b = a - (0 - b)), the same cannot be said otherwise. This basically means that **in order to complete an ISA one needs a set of instructions such that no other instruction is more powerful than the set**. 

**How do we ensure that one has a complete instruction set such that one can write any program?** The answer to this lies in finding a **Universal ISA** which would inturn constitute a **Universal Machine** which can be used to write any program known to mankind (Universal Machine has a set of basic actions where each such action can be interpretted as an instruction).

### Turing Machine 

Alan Turing, the father of computer science discovered a the theoretical device called **turing machine** which is the most powerful machine known because theoretically it can compute the results of all the programs one can be interested in.

A turing machine is a hypothetical machine which consists of an **infinite tape consisting of cells** extending in either directions, a **tape head to maintain pointer on the tape that can move left or right**, a **state cell the saves the current state** of the machine, and an **action table to write down the set of instructions**. It is posed as an thesis ( **Church-Turing Thesis** and not a theorem) that has not been counter in the past 60 years that 

> Any real-world computation can be translated into an equivalent computation involving Turing machine.

Also,

> Any computer that is equivalent to a Turing machine is said to be Turing Complete.

So the answer to **Can we build a complete ISA** lies in the question **can we design a Universal Turing Machine (UTM) that an simulate turing machine**, i.e. the all one needs to do is to build a turing machine (seemingly simple architecture) that can implement other turing machines (manage tape, tape-head, cell and action table).

So analogously speaking, the current computers are an attempt to implement this universal turing machine (UTM), where the **generic action table of the UTM is implemented as CPU**, the **the simulated action table of turing machine to be implemented is the Instruction memory**, the **working area or the UTM on the tape is the data memory**, and the **simulated state register of the implemented turing machine is the program counter (PC)**.

### Elements of Computers

- Memory (array of bytes), contains
  - program, which is a set of instructions
  - program data, i.e. variables, constants etc.

- Program Counter (PC)
  - points to an instruction the program
  - after execution of one instruction it points to the next one
  - branch instructions make PC jump to another instruction (not in sequence)

- CPU contains
  - program counter
  - instruction execution unit


### Single Instruction ISA

- sbn - subtract and branch if negative

This basically leads to the following psuedocode

```
sbn(a, b, line_no):
    a = a-b
    if (a<0):
        goto line_no
    else:
        goto next_statement
```

- Addition using SBN

```
intialize
    temp = 0
1: sbn temp, b, 2
exit: exit
2: sbn a, temp, exit
```

- Add 1-10 using SBN

```
initialize
    one = 1
    index = 10
    sum = 0

1: sbn temp, temp, 2    \\ sets temp = 0
2: sbn temp, index, 3   \\ sets temp = -index
3: sbn sum, temp, 4     \\ sets sum += index
4: sbn index, one, exit \\ sets index -= 1
5: sbn temp, temp, 6    \\ sets temp = 0
6: sbn temp, one, 1     \\ the for loop, since 0 - 1 < 0
exit: exit
```

This is similar to writing **assembly level programs**, which are low level programs.

### Mutliple Instruction ISAs

They typicall have:

- Arithmetic Instructions: Add, Subtract, Multiply, Divide
- Logical Instructions: And, Or, Not
- Move Instructions: Transfer between memory locations
- Branch Instructions: Jump to new memory locations based on program instructions

### Design of Practical Machines

- While Harvard Machine has seperate data and instruction memories, Von-Neumann Machine has a single memory to serve both the purposes.
- The problems with these machines is that they assume memory to be one large array of bytes. In practice these are slower because as the size of the structure increases the speed of processing decreases. The possible solution of this lies in having several smaller array of name locations called **registers** that can be used by instructions. Hence these smaller arrays are faster.

So,

- CPU contains a set of registers which are named storage locations.
- values are loaded from memory to registers.
- arithmetic an logical instructions use registers for input
- finally, data is stored back in the memory.

Example program in machine language,

```
r1 = mem[b] \\ load b
r2 = mem[c] \\ load c
r3 = r1 + r2 
mem[a] = r3
```

where 

- r1, r2, r3 are registers
- mem is the array of bytes representing memory

As a result the modern day computers are similar to Von-Neumann Machines with the addition of register in the CPU.

## REFERENCES:

<small>[NPTEL: Introduction to Computer Architecture](https://onlinecourses.nptel.ac.in/noc18_cs29/unit?unit=6&lesson=8){:target="_blank"}</small><br>