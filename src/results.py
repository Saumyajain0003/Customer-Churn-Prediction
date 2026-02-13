import pandas as pd
from pathlib import Path
from typing import Dict, Any

def save_results(results: Dict[str, Dict[str, Any]], out_dir: str = "src", filename: str = "results.csv") -> pd.DataFrame:
    """
    Save a dict-of-dicts `results` (model_name -> metrics dict) to CSV and JSON.
    Returns the DataFrame.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results).T
    csv_path = out / filename
    json_path = out / (Path(filename).stem + ".json")

    df.to_csv(csv_path, index=True)
    df.to_json(json_path, orient="split")

    return df