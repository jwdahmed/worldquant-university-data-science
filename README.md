# WorldQuant University – Applied Data Science (with honors)

**16-week program | Completed: December 06, 2021**

**Module I:** [Credly Badge](https://www.credly.com/badges/e1651f04-abf8-4d69-8cac-974f89abe945/public_url)
**Module II:** [Credly Badge](https://www.credly.com/badges/34f61ed4-a1b7-44c7-b378-8b9967fb3ecc/public_url)

This repository contains **code samples and project summaries** from my WQU Applied Data Science coursework. All code is original work I wrote during the program. Data and instructor materials are not included to respect WQU's intellectual property.

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

```
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

### Object-Oriented Programming (vc)

class Point:
    """2D point with vector operations"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Point(self.x * other, self.y * other)
        elif isinstance(other, Point):
            return self.x * other.x + self.y * other.y
    
    def distance(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2) ** 0.5

class Cluster:
    """K-means clustering from scratch"""
    
    def __init__(self, x, y):
        self.center = Point(x, y)
        self.points = []
    
    def update(self):
        """Recalculate centroid and reset points"""
        x_c = sum([p.x for p in self.points]) / len(self.points)
        y_c = sum([p.y for p in self.points]) / len(self.points)
        self.center = Point(x_c, y_c)
        self.points = []
```

### Data Wrangling without Pandas (`pw`)

```
from collections import defaultdict

def calc_median(vals):
    """Calculate median (handles odd/even lists)"""
    vals = sorted(vals)
    n = len(vals)
    if n % 2 == 1:
        return vals[n // 2]
    return (vals[n//2 - 1] + vals[n//2]) / 2

def calc_quantities(vals):
    """Calculate 25th, 50th, and 75th quartiles"""
    vals = sorted(vals)
    n = len(vals)
    q25 = calc_median(vals[:n//2 + 1])
    median = calc_median(vals)
    q75 = calc_median(vals[n//2:])
    return q25, median, q75

def group_by_field(data, fields):
    """Generalized grouping by any number of fields"""
    groups = defaultdict(list)
    for record in data:
        key = tuple(record[field] for field in fields)
        groups[key].append(record)
    return groups
```

### Data Wrangling with Pandas (`dw`)

```
import pandas as pd

def load_and_merge_data(scripts_path, practices_path):
    """Load and merge prescription and practice data using pandas"""
    scripts = pd.read_json(scripts_path)
    practices = pd.read_json(practices_path)
    
    # Handle duplicate postal codes (take alphabetically first)
    practice_postal = practices.groupby('code')['post_code'].min().to_dict()
    scripts['post_code'] = scripts['practice'].map(practice_postal)
    
    return scripts

def get_summary_stats(df):
    """Calculate summary statistics using pandas"""
    fields = ['items', 'quantity', 'nic', 'act_cost']
    stats = {}
    for field in fields:
        stats[field] = {
            'sum': df[field].sum(),
            'mean': df[field].mean(),
            'std': df[field].std(),
            'min': df[field].min(),
            '25%': df[field].quantile(0.25),
            '50%': df[field].median(),
            '75%': df[field].quantile(0.75),
            'max': df[field].max()
        }
    return stats

def top_drug_by_region(df):
    """Find most dispensed drug per postal code"""
    grouped = df.groupby(['post_code', 'bnf_name'])['items'].sum().reset_index()
    top_per_region = grouped.loc[grouped.groupby('post_code')['items'].idxmax()]
    region_totals = df.groupby('post_code')['items'].sum()
    top_per_region['proportion'] = top_per_region['items'] / top_per_region['post_code'].map(region_totals)
    return top_per_region.sort_values('post_code').head(100)
```

### Machine Learning (`ml`)

```
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

def build_ml_pipeline():
    """Build ML pipeline for nursing home fine prediction"""
    
    # Feature pipelines
    numeric_features = Pipeline([
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    
    categorical_features = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Classification model
    model = Pipeline([
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=50,
            random_state=42
        ))
    ])
    
    return model
```

### Natural Language Processing (`nlp`)

```
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

def build_nlp_pipeline():
    """Build NLP model for Amazon review rating prediction"""
    
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2)
        )),
        ('classifier', Ridge(alpha=3.0))
    ])
    
    return pipeline

def get_polar_words(model, vectorizer):
    """Extract most polar words from trained Naive Bayes model"""
    # This is a simplified version
    feature_names = vectorizer.get_feature_names()
    log_probs = model.feature_log_prob_
    polarity = log_probs[1] - log_probs[0]
    top_indices = np.argsort(polarity)[-25:]
    return [feature_names[i] for i in top_indices]
```
