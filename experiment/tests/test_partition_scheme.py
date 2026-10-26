"""Lock the partition-map naming contract across every entry point.

The partition map filename is assembled independently by three places:
  * the dataset-prep CLI   (experiment/prepare_dataset/get_partition.py)
  * the provisioning script (experiment/script/provision_data.sh, run_split_experiment.sh)
  * the training path       (experiment/utils/args.py + config.py -> load_partition_map)
If their defaults drift, training silently looks for a map that provisioning never
built (file-not-found), or evaluates against the wrong partitioning. These tests
assert the default scheme is `ndst_efreq_xbal` everywhere and that producer and
consumer build the *identical* filename (world size + scheme).

Dependency-free on purpose (ast + read-by-path, no dgl/torch import) so it runs on
any machine, no GPU or built extension required:
    python3 -m pytest experiment/tests/test_partition_scheme.py -v
"""
import ast
import importlib.util
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]                 # .../experiment
ARGS_PY = EXP / "utils" / "args.py"
CONFIG_PY = EXP / "utils" / "config.py"
GET_PART_PY = EXP / "prepare_dataset" / "get_partition.py"
DATALOADING_PY = EXP / "utils" / "dataloading.py"
PROVISION_SH = EXP / "script" / "provision_data.sh"
RUN_SH = EXP / "script" / "run_split_experiment.sh"

EXPECTED = {"node_weight": "dst", "edge_weight": "freq", "bal": "xbal"}
EXPECTED_TYPE = "ndst_efreq_xbal"
# The exact filename expression that must appear in BOTH producer and consumer.
NAME_TEMPLATE = 'f"{config.graph_name}_w{config.num_partition}_{config.partition_type}.npy"'


def _load_args_module():
    # args.py only imports argparse, so importing it by path is safe (no dgl/torch).
    spec = importlib.util.spec_from_file_location("groot_args", ARGS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _argparse_defaults(path):
    """{option_name: default} for every add_argument(...) in a Python file."""
    out = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            name = str(node.args[0].value).lstrip("-").replace("-", "_")
            for kw in node.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    out[name] = kw.value.value
    return out


def _config_init_default(path, field):
    """Default value of a keyword arg of Config.__init__."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            names = [a.arg for a in node.args.args]
            defaults = node.args.defaults
            offset = len(names) - len(defaults)
            for i, d in enumerate(defaults):
                if names[offset + i] == field and isinstance(d, ast.Constant):
                    return d.value
    return None


def _sh_defaults(path):
    """Parse `: "${VAR:=default}"` lines from a shell script into {VAR: default}."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(': "${') and ":=" in line:
            inner = line[len(': "${'):].split('}', 1)[0]
            var, _, default = inner.partition(":=")
            out[var] = default
    return out


def test_get_partition_type_formula():
    args = _load_args_module()
    assert args.get_partition_type("dst", "freq", "xbal") == EXPECTED_TYPE


def test_training_arg_defaults():
    d = _argparse_defaults(ARGS_PY)
    for k, v in EXPECTED.items():
        assert d.get(k) == v, f"utils/args.py --{k} default {d.get(k)!r} != {v!r}"


def test_get_partition_cli_defaults():
    d = _argparse_defaults(GET_PART_PY)
    for k, v in EXPECTED.items():
        assert d.get(k) == v, f"get_partition.py --{k} default {d.get(k)!r} != {v!r} (would break map name)"


def test_config_field_default():
    assert _config_init_default(CONFIG_PY, "partition_type") == EXPECTED_TYPE


def test_all_entrypoints_resolve_to_same_scheme():
    args = _load_args_module()
    train = _argparse_defaults(ARGS_PY)
    prep = _argparse_defaults(GET_PART_PY)
    train_type = args.get_partition_type(train["node_weight"], train["edge_weight"], train["bal"])
    prep_type = args.get_partition_type(prep["node_weight"], prep["edge_weight"], prep["bal"])
    assert train_type == prep_type == EXPECTED_TYPE


def test_shell_scripts_default_scheme():
    for sh in (PROVISION_SH, RUN_SH):
        d = _sh_defaults(sh)
        assert d.get("PART_NODE") == "dst", f"{sh.name} PART_NODE default {d.get('PART_NODE')!r}"
        assert d.get("PART_EDGE") == "freq", f"{sh.name} PART_EDGE default {d.get('PART_EDGE')!r}"
        assert d.get("PART_BAL") == "xbal", f"{sh.name} PART_BAL default {d.get('PART_BAL')!r}"


def test_map_filename_template_matches_producer_and_consumer():
    # World size (num_partition) + scheme (partition_type) must be the identical
    # expression on both sides, so a provisioned map is exactly what training loads.
    assert NAME_TEMPLATE in GET_PART_PY.read_text(), "producer map-name template changed"
    assert NAME_TEMPLATE in DATALOADING_PY.read_text(), "consumer map-name template changed"
