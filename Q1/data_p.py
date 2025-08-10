import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pandas as pd


class MovieStatsManager:
    def __init__(self, db_path='movie_stats.json'):
        self.db_path = Path(db_path)
        self.CATEGORIES = {
            'actors': 'cast',
            'directors': 'director',
            'writers': 'writers',
            'genres': 'genres',
            'producers': 'producers'
        }
        self.stats = self._init_db()

    def _init_db(self):
        def nested_defaultdict():
            return defaultdict(lambda: {"count": 0, "total_rating": 0.0, "average_rating": 0.0})

        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                db = json.load(f)
                for category in self.CATEGORIES:
                    if category not in db:
                        db[category] = nested_defaultdict()
                    else:
                        db[category] = defaultdict(
                            lambda: {"count": 0, "total_rating": 0.0, "average_rating": 0.0},
                            db[category]
                        )
                return db
        return {
            category: nested_defaultdict()
            for category in self.CATEGORIES
        } | {"metadata": {"last_updated": datetime.now().isoformat()}}

    def _process_multi_value(self, raw_value):
        if pd.isna(raw_value) or raw_value == 'Unknown':
            return []
        return [item.strip() for item in str(raw_value).split(',')]

    def process_row(self, row):
        for category, column in self.CATEGORIES.items():
            items = self._process_multi_value(row[column])
            for item in items:
                # 现在可以安全访问，即使item不存在也会自动初始化
                self.stats[category][item]["count"] += 1
                self.stats[category][item]["total_rating"] += row['rating']
                self.stats[category][item]["average_rating"] = (
                        self.stats[category][item]["total_rating"] /
                        self.stats[category][item]["count"]
                )

    def save(self):
        self.stats["metadata"]["last_updated"] = datetime.now().isoformat()
        # 保存前将defaultdict转换为普通dict
        save_stats = {
            category: dict(self.stats[category])
            for category in self.CATEGORIES
        }
        save_stats["metadata"] = self.stats["metadata"]
        with open(self.db_path, 'w') as f:
            json.dump(save_stats, f, indent=2)

    def get_stats(self, category, name):
        return self.stats[category].get(name, {"count": 0, "total_rating": 0.0, "average_rating": 0.0})

def process_dataframe(df, manager):
    for _, row in df.iterrows():
        manager.process_row(row)
    manager.save()

# 使用示例
df = pd.read_csv('../df_movies_cleaned.csv')
manager = MovieStatsManager()
process_dataframe(df, manager)