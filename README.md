# CS4100 Project Proposal

## Neil Agrawal, Evelyn Robert, Kelsey Nihezagirwe
## Overview

We propose to build a fully automatic crossword-puzzle generator that, given a curated list of words and their human-written clues, outputs a complete American-style crossword grid that satisfies symmetry, connectivity, and high fill density. The system will rely solely on classic AI search and constraint-satisfaction techniques, not large language models.

## Motivation

Creating a high-quality crossword puzzle takes planning and thoughtful word choice. Smaller organizations—like schools or local papers—often don’t have the time, tools, or resources to build custom puzzles that fit their specific needs. An open-source AI tool that automates crossword layout could make puzzle creation more accessible, allowing educators and creators to easily generate personalized content like vocabulary reviews. It would also serve as a hands-on example of how concepts like constraint satisfaction and heuristic search apply to real-world problems. These can be done using CSP and heuristic search. Each word slot in the grid is treated as a variable, and valid words from a dictionary form the domain of those variables. Constraints ensure that words intersect correctly and fit within the specified grid dimensions.

## Related Work

[**cwc Crossword Compiler**](https://cwordc.sourceforge.net/)   
Lars Christensen, 1999‑2002

An open‑source C++ constructor that fills a pre‑defined grid by **exhaustive letter‑by‑letter backtracking search** with dependency‑aware backtrack pruning and dictionary‑level optimisations; it focuses on speed and correctness of fill but leaves clue writing entirely to humans.

[**Quick Generation of Crosswords Using Concatenation**](https://ieee-cog.org/2022/assets/papers/paper_194.pdf)  
Dakowski, Jaworski & Wojna, 2024

Proposes an inductive approach that concatenates, rotates, and mutates smaller sub‑crosswords. Two variants: the first is **first‑/best‑improvement local search** and the second is **simulated annealing**, optimizing a cost function combining intersection count and letter density. Resulting puzzles meet structural constraints but are often too difficult for human solvers. The authors suggest future work on refined fitness metrics and cuckoo‑search.

[**A Fully Automatic Crossword Generator**](https://www.researchgate.net/publication/224362845_A_Fully_Automatic_Crossword_Generator)  
Rigutini, Leonardo & Diligenti, Michelangelo & Maggini, Marco & Gori, Marco (2009)

Frames crossword generation as a **constraint satisfaction problem (CSP)**, where word placement is optimized under structural and lexical constraints. Their system also includes an **NLP-based clue extraction module** that automatically gathers definitions from web sources, enabling fully automated puzzle creation.

**Disclaimer:**

LLMs were used to refine content, build upon initial structure, improve writing clarity/wording, and to find sources (which we vetted ourselves)

