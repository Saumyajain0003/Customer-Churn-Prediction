import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def save_results(
    results: Dict[str, Dict[str, Any]],
    out_dir: str | Path = "results",
    filename: str = "results.csv",
) -> pd.DataFrame:
    """
    Save a dict-of-dicts `results` (model_name → metrics dict) to CSV and JSON.

    Args:
        results:  {model_name: {metric: value, ...}, ...}
        out_dir:  Output directory (default: 'results/').
        filename: Base filename for CSV (JSON uses same stem).

    Returns:
        DataFrame of results (models as rows, metrics as columns).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results).T
    df.index.name = "model"

    csv_path = out / filename
    json_path = out / (Path(filename).stem + ".json")

    df.to_csv(csv_path, index=True)
    df.to_json(json_path, orient="split", indent=2)

    logger.info(f"Results saved → {csv_path.resolve()}")
    logger.info(f"Results saved → {json_path.resolve()}")

    # Pretty-print summary table
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(df.to_string(float_format=lambda x: f"{x:.4f}" if x is not None else "N/A"))
    print("=" * 60)

    return df