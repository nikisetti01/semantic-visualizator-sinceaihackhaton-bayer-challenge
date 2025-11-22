# models/category_stats.py

from dataclasses import dataclass

@dataclass
class CategoryStats:
    dimension_type: str
    category: str
    support_count: int
    support_ratio: float
    mean_score: float
