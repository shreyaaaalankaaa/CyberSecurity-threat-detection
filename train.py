from pathlib import Path
import json
from src.data import generate_demo_dataset, validate_dataset
from src.modeling import train_and_compare, save_model


def main() -> None:
    df = validate_dataset(generate_demo_dataset())
    results = train_and_compare(df)
    best = results[0]
    model_path = save_model(best)
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/metrics.json").write_text(json.dumps({r.model_name: r.metrics for r in results}, indent=2))
    print(f"Saved {best.model_name} to {model_path}")
    print(json.dumps(best.metrics, indent=2))

if __name__ == "__main__":
    main()
