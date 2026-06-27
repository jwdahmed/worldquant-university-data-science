# WorldQuant University – Applied Data Science I (with honors)

**8-week program | Completed: September 11, 2021**
**Credential:** [Credly Badge](https://www.credly.com/badges/e1651f04-abf8-4d69-8cac-974f89abe945/public_url)

This repository contains **code samples and project summaries** from my WQU Applied Data Science I coursework. All code is original work I wrote during the program. Data and instructor materials are not included to respect WQU's intellectual property.

---

## Course Modules & Skills

| Module | Focus | Key Skills Demonstrated |
|--------|-------|------------------------|
| **ip** (Program Flow) | Iteration & conditional logic | Mersenne numbers, Lucas-Lehmer primality test, Sieve of Eratosthenes |
| **vc** (Object-Oriented Programming) | Class design & OOP | Built `Point` class with operator overloading; implemented K-means clustering |
| **pw** (Pure Python Data Wrangling) | Data structures & functions | Built `describe()`, `group_by_field()`; analyzed 382k NHS records without pandas |
| **dw** (Data Wrangling with Pandas) | Tabular data manipulation | Applied pandas for efficient grouping, merging, and aggregation |
| **ml** (Machine Learning) | Predictive modeling | Built classification models to predict nursing home fines |
| **nlp** (Natural Language Processing) | Text classification | Trained models to predict Amazon review ratings from text |

---

## Key Functions & Algorithms (Pure Python)

### Program Flow (`ip`)

```python
def mersenne_number(p):
    """Return Mersenne number: 2^p - 1"""
    return (2 ** p) - 1

def lucas_lehmer(p):
    """Generate Lucas-Lehmer sequence for Mersenne prime testing"""
    sequence = [4]
    for i in range(1, p-1):
        sequence.append((sequence[i-1]**2 - 2) % (2**p - 1))
    return sequence

def is_prime_fast(n):
    """Optimized primality check (square root optimization)"""
    if n <= 1: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def sieve(n):
    """Sieve of Eratosthenes (dynamic programming approach)"""
    bool_list = [True] * (n + 1)
    bool_list[0] = bool_list[1] = False
    for i in range(2, int(n**0.5) + 1):
        if bool_list[i]:
            for j in range(i*i, n+1, i):
                bool_list[j] = False
    return [i for i, val in enumerate(bool_list) if val]
