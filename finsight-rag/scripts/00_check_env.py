# %% [markdown]
# # Step 00 — Check the environment   (Phase P0)
#
# **What this step does:** loads the layered config (`base.yaml` + profile + `.env` + environment)
# and checks everything FinSight needs: Python 3.12, PyTorch with the Apple GPU (MPS), key
# packages, data/index folders, the local Ollama server and model, and (optionally) API keys.
#
# **Three ways to run it**
# - Terminal, all at once: `uv run python scripts/00_check_env.py [--config configs/cloud.yaml]`
# - VS Code, cell by cell: click **Run Cell** above each `# %%` (the Interactive Window keeps
#   every variable)
# - Jupyter: `make notebooks`, then open `notebooks/00_check_env.ipynb`
#
# **What to look at:** `fail` rows must be fixed; `warn` = optional feature off (e.g. Ollama,
# needed from P6); `info` = not needed yet. Then see how the config **fingerprint** reacts to a
# changed setting.

# %%
import pandas as pd

from finsight.config import ENV_ALIASES, load_settings
from finsight.container import Container
from finsight.diagnostics import run_all_checks
from finsight.notebook import Param, configure_pandas, in_notebook, script_args, show, step

configure_pandas()

# %% Parameters: in a notebook, edit the defaults here; in the terminal, use the flags.
P = script_args(
    "Step 00: check the environment",
    config=Param(None, str, "profile YAML (default: configs/local.yaml)"),
)
P

# %% [markdown]
# ## 1. Load the layered configuration
# `load_settings()` merges `configs/base.yaml` ← profile ← `.env`/environment ← overrides,
# validates every value, and resolves paths against the project root.

# %%
step("1. Load the layered configuration")
settings = load_settings(P.config)
show(
    pd.Series(
        {
            "project root": str(settings.root),
            "profile": str(settings.config_path.relative_to(settings.root)),
            "fingerprint": settings.fingerprint(),
            "env aliases read": ", ".join(ENV_ALIASES) + ", plus any FINSIGHT__SECTION__KEY",
        }
    ),
    "Resolved config",
)

# %% [markdown]
# ## 2. Run every environment check
# Each check returns a `CheckResult(name, status, detail, hint)`. Here they become a DataFrame.

# %%
step("2. Run every environment check")
results = run_all_checks(settings)
checks_df = pd.DataFrame([r.__dict__ for r in results])
show(checks_df, "Environment checks")
show(checks_df["status"].value_counts(), "Status counts")

# %% [markdown]
# ## 3. What would the pipeline build with this config?
# The `Container` builds every component from one `Settings` object, so the API, the scripts
# and the evaluation always use the same pipeline.

# %%
step("3. What the pipeline would build")
show(pd.Series(Container(settings).describe()), "Container.describe()")

# %% [markdown]
# ## 4. Experiment: how the fingerprint tracks settings
# Change a setting that affects results (chunk size) and one that doesn't (log level). Only the
# first changes the fingerprint. That's how every index and eval result stays traceable to its
# config.

# %%
step("4. Fingerprint experiment")
variants = {
    "as loaded": settings,
    "chunk_tokens=256": load_settings(P.config, {"chunking": {"chunk_tokens": 256}}),
    "logging.level=DEBUG": load_settings(P.config, {"logging": {"level": "DEBUG"}}),
}
fingerprints_df = pd.DataFrame(
    [
        {
            "variant": name,
            "fingerprint": s.fingerprint(),
            "same as loaded": s.fingerprint() == settings.fingerprint(),
        }
        for name, s in variants.items()
    ]
)
show(fingerprints_df, "Fingerprints")

# %% Summary
step("Summary")
n_fail = int((checks_df["status"] == "fail").sum())
if n_fail:
    print(f"{n_fail} failing check(s): fix them (see the 'hint' column), then run again.")
    if not in_notebook():
        raise SystemExit(1)
else:
    print("Ready for P1: uv run python scripts/01_download_data.py")
