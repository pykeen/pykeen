import json
import pathlib
import yaml

if __name__ == "__main__":
    root = pathlib.Path("./src/pykeen/experiments/")
    for path in root.rglob("*.json"):
        with path.open() as f:
            c = json.load(f)
        with path.with_suffix(".yaml").open("w") as f:
            yaml.dump(c, f)
        path.unlink()
