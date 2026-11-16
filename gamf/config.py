"""YAML configuration loading with ``key.sub=value`` command-line overrides.

Several YAML files are merged left to right (e.g. base -> dataset -> model),
so every experiment is fully described by the files it was started with.
The merged configuration is written to each run directory for provenance.
"""
import ast
import copy
from pathlib import Path

import yaml


def _merge(dst, src):
    for key, val in src.items():
        if isinstance(val, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], val)
        else:
            dst[key] = copy.deepcopy(val)
    return dst


def _parse_value(text):
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def load_config(paths, overrides=()):
    cfg = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    for item in overrides or ():
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_value(val)
    return cfg


def save_config(cfg, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def flatten(cfg, prefix=""):
    """Nested configuration -> ordered list of (dotted key, value)."""
    rows = []
    for key, val in cfg.items():
        name = f"{prefix}{key}"
        if isinstance(val, dict):
            rows += flatten(val, name + ".")
        else:
            rows.append((name, val))
    return rows


def save_hparam_table(cfg, out_dir, extra=None):
    """Store every hyperparameter of a run as a table: hyperparameters.csv (machine
    readable) and hyperparameters.md (human readable), next to config.yaml."""
    import csv
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = flatten(cfg) + sorted((extra or {}).items())
    with open(out_dir / "hyperparameters.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value"])
        w.writerows(rows)
    lines = ["| Parameter | Value |", "|---|---|"] + [f"| `{k}` | {v} |" for k, v in rows]
    (out_dir / "hyperparameters.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_config_args(parser):
    parser.add_argument("--config", nargs="+", required=True,
                        help="YAML files merged left to right")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="overrides, e.g. train.batch_size=8 seed=1")
    return parser


def dataset_dir(cfg):
    return Path(cfg["paths"]["data_root"]) / cfg["data"]["name"]


def run_dir(cfg):
    return (Path(cfg["paths"]["runs_root"]) / cfg["data"]["name"]
            / cfg["model"]["name"] / f"seed_{cfg['seed']}")


def damsm_path(cfg):
    return Path(cfg["paths"]["runs_root"]) / cfg["data"]["name"] / "damsm" / "damsm_best.pt"
