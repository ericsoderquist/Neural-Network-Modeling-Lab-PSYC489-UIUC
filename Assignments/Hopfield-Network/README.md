# Simple Hopfield Network: A Comprehensive Investigation for Neural Network Modeling Lab PSYC 489
## University of Illinois Urbana-Champaign
### Author: Eric Soderquist

---

## Table of Contents
1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [Prerequisites](#prerequisites)
4. [File Structure](#file-structure)
5. [Methods Overview](#methods-overview)
6. [Usage](#usage)
7. [Theoretical Background](#theoretical-background)
8. [Performance Metrics](#performance-metrics)
9. [Contributions](#contributions)
10. [License](#license)

---

## Introduction

This repository contains a rigorously developed and optimized implementation of a Simple Hopfield Network. This project aligns with the academic goals of the Neural Network Modeling Lab course (PSYC 489) at the University of Illinois Urbana-Champaign but extends beyond the coursework to explore best practices in machine learning and software engineering.

---

## Motivation

The Simple Hopfield Network has been a subject of interest in both the fields of psychology and computer science due to its applications in associative memory and optimization problems. This implementation adheres to the highest standards of academic rigor and software engineering.

---

## Prerequisites

- Python 3.8+
- NumPy

---

## File Structure

- `Simple_Hopfield_Network_Updated_Further.py`: The Python script containing the optimized Hopfield Network implementation.
    - Methods: `__init__`, `train`, `recall`

---

## Methods Overview

- `__init__`: Initializes the Hopfield Network with error handling for input validation.
- `train`: Trains the network on a given set of patterns, incorporating input validation and error handling.
- `recall`: Recalls stored patterns based on partial inputs. Input validation and error handling are also included.

---

## Usage

To run the code, execute the following command:

```bash
python Simple_Hopfield_Network_Updated_Further.py
```

---

## Theoretical Background

The Hopfield Network serves as an associative memory system with binary threshold nodes. The network is trained using the Hebbian learning rule and can recall stored patterns when provided with partial patterns.

---

## Performance Metrics

- Hamming Distance: Measures the difference between the recalled pattern and the original pattern.
- Recall Success Rate: The proportion of successfully recalled patterns over a set of trials.

---

## Contributions

This project was solely developed by Eric Soderquist, leveraging a deep understanding of machine learning algorithms and software engineering principles.

---

## License

This project is licensed under the MIT License. For more details, please see the [LICENSE](LICENSE.md) file.

