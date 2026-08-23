import os
import yaml

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PKG_DIR)
ENV_VAR = "IRONPULSE_PROJECT_ROOT"
LEGACY_ROOT = os.path.join(REPO_DIR, "research")


def find_project_root(start=None):
    env = os.environ.get(ENV_VAR)
    if env and os.path.isfile(os.path.join(env, "config.yaml")):
        return os.path.abspath(env)
    here = os.path.abspath(start or os.getcwd())
    while True:
        if os.path.isfile(os.path.join(here, "config.yaml")):
            return here
        nxt = os.path.dirname(here)
        if nxt == here:
            break
        here = nxt
    if os.path.isfile(os.path.join(LEGACY_ROOT, "config.yaml")):
        return LEGACY_ROOT
    raise RuntimeError(
        "cannot locate a project root containing config.yaml; "
        "set %s to the project directory" % ENV_VAR)


ROOT = find_project_root()


def load(path=None, root=None):
    r = os.path.abspath(root) if root else ROOT
    p = path or os.path.join(r, "config.yaml")
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = r
    for k, v in cfg.get("paths", {}).items():
        cfg["paths"][k] = os.path.join(r, v)
    return cfg


CFG = load()


def path(key, *parts):
    return os.path.join(CFG.get("paths", {})[key], *parts)


def ensure(p):
    os.makedirs(p, exist_ok=True)
    return p
