import json
import argparse
import time
from src.pipeline import Pipeline


def main(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    pipeline = Pipeline()
    results = []

    for item in data:
        start = time.time()

        preds = pipeline.run(item["query"])

        latency = time.time() - start

        results.append({
            "id": item["id"],
            "retrieved_standards": preds,
            "latency_seconds": round(latency, 3)
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    main(args.input, args.output)