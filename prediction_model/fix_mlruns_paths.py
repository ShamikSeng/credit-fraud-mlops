"""
Rewrites the host-absolute `file://` paths MLflow's local FileStore bakes into every
experiment/run `meta.yaml` (artifact_location / artifact_uri) into the portable relative
form `file:./mlruns/...`.

Why this exists: `config.TRACKING_URI` is the relative URI `file:./mlruns`, but MLflow's
FileStore resolves that to an ABSOLUTE path (based on the current working directory) at
experiment/run *creation* time and stores that absolute path permanently in each run's
meta.yaml. On this machine that's an absolute Windows host path
(`file:///C:/Users/.../mlruns/...`). Phase 3c mounts the same `mlruns/` directory into the
Docker container at `/app/mlruns` so predictions can load a real trained model without
baking one into the image — but the container's filesystem root isn't `C:/Users/...`, so
`mlflow.sklearn.load_model` fails with `OSError: No such file or directory`. Rewriting the
baked-in prefix to the same `file:./mlruns` relative form the tracking URI already uses
makes MLflow resolve it relative to the process's cwd instead — which is the repo root on
the host and `/app` in the container, so the same directory works in both.

This has to be re-run after every new training run (each new run bakes in a fresh absolute
path), so it's a documented prerequisite before `docker compose up` / `docker run`, the same
tier as `python -m prediction_model.monitoring.make_reference`. Run:
    python -m prediction_model.fix_mlruns_paths
"""
import glob
import os
import re

from prediction_model.config import config

MLRUNS_DIR = os.path.join(os.path.dirname(config.PACKAGE_ROOT), "mlruns")


def fix_mlruns_paths(mlruns_dir=MLRUNS_DIR):
    """Rewrite every meta.yaml's baked absolute-path file:// URI to a relative one anchored
    at `mlruns_dir`'s own name. Returns the number of files changed."""
    # Normalize to forward slashes BEFORE escaping - escaping first then replacing would
    # mangle the doubled backslashes re.escape produces on Windows paths.
    abs_path_forward = os.path.abspath(mlruns_dir).replace("\\", "/")
    abs_prefix_pattern = re.compile(r"file:///?" + re.escape(abs_path_forward))
    changed = 0
    for meta_path in glob.glob(os.path.join(mlruns_dir, "**", "meta.yaml"), recursive=True):
        with open(meta_path, "r", encoding="utf-8") as f:
            content = f.read()
        new_content = abs_prefix_pattern.sub("file:./mlruns", content)
        if new_content != content:
            with open(meta_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            changed += 1
    return changed


if __name__ == "__main__":
    n = fix_mlruns_paths()
    print(f"Rewrote {n} meta.yaml file(s) under {MLRUNS_DIR} to use relative artifact paths.")
