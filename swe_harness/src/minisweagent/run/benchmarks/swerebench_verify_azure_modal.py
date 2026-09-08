#!/usr/bin/env python3

"""Verify patches for SWE-rebench instances using Azure Modal sandbox.

Reads patches from trajectory files in an output directory, then evaluates
each patch by running tests inside an Azure Modal sandbox.

Follows the SWE-rebench evaluation approach from:
https://github.com/SWE-rebench/SWE-bench-fork
- Applies model patch via git apply (with fallbacks)
- Runs eval script: conda activate testbed (for Python repos), reset tests,
  apply test_patch via heredoc, START/END markers around test command
- Parses output locally using inlined log parsers from the SWE-rebench fork
"""

import base64
import concurrent.futures
import json
import os
import re
import shlex
import threading
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live

from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
from minisweagent.utils.log import add_file_handler, logger
from minisweagent.utils.serialize import UNSET, recursive_merge

console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich", add_completion=False)
_OUTPUT_FILE_LOCK = threading.Lock()
_RESULTS_CACHE: dict | None = None
_RESULTS_DIRTY = False

DATASET_MAPPING = {
    "rebench": "nebius/SWE-rebench",
    "rebench_v2": "nebius/SWE-rebench-V2",
}

DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench_azure_modal.yaml"

# From SWE-rebench fork constants
START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
RESET_FAILED = ">>>>> Reset Failed"
TESTS_ERROR = ">>>>> Tests Errored"
TESTS_TIMEOUT = ">>>>> Tests Timed Out"
DOCKER_WORKDIR = "/testbed"


# Repos where only FAIL_TO_PASS tests are checked (no PASS_TO_PASS)
# From SWE-rebench fork constants/__init__.py FAIL_ONLY_REPOS
_FAIL_ONLY_REPOS = {"chartjs/Chart.js", "processing/p5.js", "markedjs/marked"}

# Non-test extensions (from SWE-rebench fork NON_TEST_EXTS)
_NON_TEST_EXTS = [".json", ".png", "csv", ".txt", ".md", ".jpg", ".jpeg", ".pkl",
                  ".zip", ".csv", ".parquet", ".feather", ".svg", ".npy", ".yml", ".yaml"]

_TIMING_NORMALIZE_RES = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Inlined log parsers from SWE-rebench fork swebench/harness/log_parsers/
# Each parser takes (log: str) and returns {test_name: status_str}
# Status strings: "PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL"
# ---------------------------------------------------------------------------

def _ansi_escape(text: str) -> str:
    return re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])").sub("", text)


# -- Python parsers --

def _parse_log_pytest(log: str) -> dict[str, str]:
    sm = {}
    for line in log.split("\n"):
        if any(line.startswith(s) for s in ["PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL"]):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) >= 2:
                sm[parts[1]] = parts[0]
    return sm


def _parse_log_pytest_options(log: str) -> dict[str, str]:
    opt_re = re.compile(r"(.*?)\[(.*)\]")
    sm = {}
    for line in log.split("\n"):
        if any(line.startswith(s) for s in ["PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL"]):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) >= 2:
                m = opt_re.search(parts[1])
                if m:
                    main, opt = m.groups()
                    if opt.startswith("/") and not opt.startswith("//") and "*" not in opt:
                        opt = "/" + opt.split("/")[-1]
                    name = f"{main}[{opt}]"
                else:
                    name = parts[1]
                sm[name] = parts[0]
    return sm


def _parse_log_pytest_v2(log: str) -> dict[str, str]:
    sm = {}
    escapes = "".join(chr(c) for c in range(1, 32))
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line).translate(str.maketrans("", "", escapes))
        if any(line.startswith(s) for s in ["PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL"]):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) >= 2:
                sm[parts[1]] = parts[0]
        elif any(line.endswith(s) for s in ["PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL"]):
            parts = line.split()
            if len(parts) >= 2:
                sm[parts[0]] = parts[1]
    return sm


def _parse_log_django(log: str) -> dict[str, str]:
    sm = {}
    prev_test = None
    for line in log.split("\n"):
        line = line.strip()
        if "--version is equivalent to version" in line:
            sm["--version is equivalent to version"] = "PASSED"
        if " ... " in line:
            prev_test = line.split(" ... ")[0]
        for suffix in (" ... ok", " ... OK", " ...  OK"):
            if line.endswith(suffix):
                sm[line.rsplit(suffix, 1)[0]] = "PASSED"
                break
        if " ... skipped" in line:
            sm[line.split(" ... skipped")[0]] = "SKIPPED"
        if line.endswith(" ... FAIL"):
            sm[line.split(" ... FAIL")[0]] = "FAILED"
        if line.startswith("FAIL:"):
            sm[line.split()[1].strip()] = "FAILED"
        if line.endswith(" ... ERROR"):
            sm[line.split(" ... ERROR")[0]] = "ERROR"
        if line.startswith("ERROR:"):
            sm[line.split()[1].strip()] = "ERROR"
        if line.lstrip().startswith("ok") and prev_test is not None:
            sm[prev_test] = "PASSED"
    return sm


def _parse_log_seaborn(log: str) -> dict[str, str]:
    sm = {}
    for line in log.split("\n"):
        if line.startswith("FAILED"):
            parts = line.split()
            if len(parts) >= 2:
                sm[parts[1]] = "FAILED"
        elif " PASSED " in line:
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "PASSED":
                sm[parts[0]] = "PASSED"
        elif line.startswith("PASSED"):
            parts = line.split()
            if len(parts) >= 2:
                sm[parts[1]] = "PASSED"
    return sm


def _parse_log_sympy(log: str) -> dict[str, str]:
    sm = {}
    for match in re.findall(r"(_*) (.*)\.py:(.*) (_*)", log):
        sm[f"{match[1]}.py:{match[2]}"] = "FAILED"
    for line in log.split("\n"):
        line = line.strip()
        if line.startswith("test_"):
            if line.endswith(" E"):
                sm[line.split()[0]] = "ERROR"
            elif line.endswith(" F"):
                sm[line.split()[0]] = "FAILED"
            elif line.endswith(" ok"):
                sm[line.split()[0]] = "PASSED"
    return sm


def _parse_log_matplotlib(log: str) -> dict[str, str]:
    log = log.replace("MouseButton.LEFT", "1").replace("MouseButton.RIGHT", "3")
    return _parse_log_pytest(log)


# -- Go parsers --

def _parse_log_gotest(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^--- (PASS|FAIL|SKIP): (.+) \(.+\)$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            status, name = m.groups()
            sm[name] = {"PASS": "PASSED", "FAIL": "FAILED", "SKIP": "SKIPPED"}[status]
    return sm


# -- Java parsers --

def _parse_log_maven(log: str) -> dict[str, str]:
    sm = {}
    current = "---NO TEST NAME FOUND YET---"
    name_pat = re.compile(r"^.*-Dtest=(\S+).*$")
    result_pat = re.compile(r"^.*BUILD (SUCCESS|FAILURE)$")
    for line in log.split("\n"):
        m = name_pat.match(line.strip())
        if m:
            current = m.group(1)
        m = result_pat.match(line.strip())
        if m:
            sm[current] = "PASSED" if m.group(1) == "SUCCESS" else "FAILED"
    return sm


def _parse_log_ant(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^\s*\[junit\]\s+\[(PASS|FAIL|ERR)\]\s+(.*)$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            status, name = m.groups()
            sm[name] = "PASSED" if status == "PASS" else "FAILED"
    return sm


def _parse_log_gradle(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^([^>].+)\s+(PASSED|FAILED)$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            name, status = m.groups()
            sm[name] = status
    return sm


# -- JavaScript parsers --

def _parse_log_jest(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^\s*(✓|✕|○)\s(.+?)(?:\s\((\d+\s*m?s)\))?$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            sym, name, _ = m.groups()
            sm[name] = {"✓": "PASSED", "✕": "FAILED", "○": "SKIPPED"}[sym]
    return sm


def _parse_log_jest_json(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^\[(PASSED|FAILED)\]\s(.+)$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            status, name = m.groups()
            sm[name] = status
    return sm


def _parse_log_vitest(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^\s*(✓|×|↓)\s(.+?)(?:\s(\d+\s*m?s?|\[skipped\]))?$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            sym, name, _ = m.groups()
            sm[name] = {"✓": "PASSED", "×": "FAILED", "↓": "SKIPPED"}[sym]
    return sm


def _parse_log_tap(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^(ok|not ok) (\d+) (.+)$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            status, _, name = m.groups()
            sm[name] = "PASSED" if status == "ok" else "FAILED"
    return sm


def _parse_log_marked(log: str) -> dict[str, str]:
    sm = {}
    for line in log.split("\n"):
        m = re.search(r"^\d+\)\s(.*)", line)
        if m:
            sm[m.group(1).strip()] = "FAILED"
    return sm


def _parse_log_react_pdf(log: str) -> dict[str, str]:
    sm = {}
    for line in log.split("\n"):
        for pattern, status in [
            (r"^PASS\s(.*)\s\([\d\.]+ ?[ms]+\)", "PASSED"),
            (r"^PASS\s(.*)", "PASSED"),
            (r"^FAIL\s(.*)\s\([\d\.]+ ?[ms]+\)", "FAILED"),
            (r"^FAIL\s(.*)", "FAILED"),
        ]:
            m = re.match(pattern, line)
            if m:
                sm[m.group(1)] = status
                break
    return sm


# -- PHP parsers --

def _parse_log_phpunit(log: str) -> dict[str, str]:
    sm = {}
    suite = None
    suite_pat = re.compile(r"^(\w.+) \(.+\)$")
    test_pat = re.compile(r"^\s*([✔✘↩])\s*(.*)$")
    for line in log.split("\n"):
        m = suite_pat.match(line)
        if m:
            suite = m.group(1)
            continue
        m = test_pat.match(line)
        if m:
            sym, name = m.groups()
            full = f"{suite} > {name}"
            sm[full] = {"✔": "PASSED", "✘": "FAILED", "↩": "SKIPPED"}[sym]
    return sm


# -- Ruby parsers --

def _parse_log_ruby_unit(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^\s*(?:test: )?(.+):\s+(\.|E\b|F\b|O\b)")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            name, outcome = m.groups()
            sm[name] = {".": "PASSED", "E": "FAILED", "F": "FAILED", "O": "SKIPPED"}[outcome]
    return sm


def _parse_log_rspec_json(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"(.+) - (passed|failed|pending)")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            name, outcome = m.groups()
            sm[name] = {"passed": "PASSED", "failed": "FAILED", "pending": "SKIPPED"}[outcome]
    return sm


def _parse_log_minitest(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^(.+)\. .*=.*(\.|F|E).*$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            name, outcome = m.groups()
            sm[name] = {".": "PASSED", "F": "FAILED", "E": "FAILED"}[outcome]
    return sm


def _parse_log_jekyll(log: str, instance_id: str) -> dict[str, str]:
    pr = instance_id.split("-")[1]
    if pr in ["9141", "8047", "8167"]:
        return _parse_log_minitest(log)
    elif pr in ["8761", "8771"]:
        # cucumber - simplified
        sm = {}
        pat = re.compile(r"^(.*) \.+(\.|F)")
        for line in log.split("\n"):
            m = pat.match(line.strip())
            if m:
                name, outcome = m.groups()
                sm[name] = "PASSED" if outcome == "." else "FAILED"
        return sm
    return {}


# -- Rust parsers --

def _parse_log_cargo(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^test\s+(\S+)\s+\.\.\.\s+(\w+)$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            name, outcome = m.groups()
            if outcome == "ok":
                sm[name] = "PASSED"
            elif outcome == "FAILED":
                sm[name] = "FAILED"
    return sm


# -- C parsers --

def _parse_log_redis(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^\[(ok|err|skip|ignore)\]:\s(.+?)(?:\s\([\d\s\w]+\))?$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            status, name = m.groups()
            if status == "ok":
                sm[name] = "PASSED"
            elif status == "err":
                name = re.sub(r"\s+in\s+\S+$", "", name)
                sm[name] = "FAILED"
            elif status in ("skip", "ignore"):
                sm[name] = "SKIPPED"
    return sm


def _parse_log_jq(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^\s*(PASS|FAIL):\s(.+)$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            status, name = m.groups()
            sm[name] = "PASSED" if status == "PASS" else "FAILED"
    return sm


def _parse_log_doctest(log: str) -> dict[str, str]:
    sm = {}
    start = log.find("<doctest")
    end_tag = "</doctest>"
    end = log.find(end_tag, start) + len(end_tag) if start != -1 else -1
    if start != -1 and end != -1:
        try:
            root = ET.fromstring(log[start:end])
            for tc in root.findall(".//TestCase"):
                tc_name = tc.get("name")
                for sc in tc.findall(".//SubCase"):
                    sc_name = sc.get("name")
                    name = f"{tc_name} > {sc_name}"
                    exprs = sc.findall(".//Expression")
                    sm[name] = "PASSED" if all(e.get("success") == "true" for e in exprs) else "FAILED"
        except ET.ParseError:
            pass
    return sm


def _parse_log_micropython(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^(pass|FAIL|skip)\s+(.+)$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            status, name = m.groups()
            sm[name] = {"pass": "PASSED", "FAIL": "FAILED", "skip": "SKIPPED"}[status]
    return sm


def _parse_log_googletest(log: str) -> dict[str, str]:
    sm = {}
    pat = re.compile(r"^.*\[\s*(OK|FAILED)\s*\]\s(.*)\s\(.*\)$")
    for line in log.split("\n"):
        m = pat.match(line.strip())
        if m:
            status, name = m.groups()
            sm[name] = "PASSED" if status == "OK" else "FAILED"
    return sm


# Map log_parser name (from install_config['log_parser']) -> parser function
# Matches MAP_REPO_TO_PARSER in SWE-rebench fork (repo names + parser name strings)
_LOG_PARSERS: dict[str, object] = {
    # by parser name (install_config['log_parser'] value)
    "parse_log_pytest": _parse_log_pytest,
    "parse_log_pytest_options": _parse_log_pytest_options,
    "parse_log_pytest_v2": _parse_log_pytest_v2,
    "parse_log_django": _parse_log_django,
    "parse_log_seaborn": _parse_log_seaborn,
    "parse_log_sympy": _parse_log_sympy,
    "parse_log_matplotlib": _parse_log_matplotlib,
    "parse_log_gotest": _parse_log_gotest,
    "parse_log_maven": _parse_log_maven,
    "parse_log_ant": _parse_log_ant,
    "parse_log_gradle_custom": _parse_log_gradle,
    "parse_log_jest": _parse_log_jest,
    "parse_log_jest_json": _parse_log_jest_json,
    "parse_log_vitest": _parse_log_vitest,
    "parse_log_tap": _parse_log_tap,
    "parse_log_marked": _parse_log_marked,
    "parse_log_react_pdf": _parse_log_react_pdf,
    "parse_log_phpunit": _parse_log_phpunit,
    "parse_log_ruby_unit": _parse_log_ruby_unit,
    "parse_log_rspec_transformed_json": _parse_log_rspec_json,
    "parse_log_cargo": _parse_log_cargo,
    "parse_log_redis": _parse_log_redis,
    "parse_log_jq": _parse_log_jq,
    "parse_log_doctest": _parse_log_doctest,
    "parse_log_micropython_test": _parse_log_micropython,
    "parse_log_googletest": _parse_log_googletest,
    # by repo name (MAP_REPO_TO_PARSER_PY)
    "astropy/astropy": _parse_log_pytest_v2,
    "django/django": _parse_log_django,
    "marshmallow-code/marshmallow": _parse_log_pytest,
    "matplotlib/matplotlib": _parse_log_matplotlib,
    "mwaskom/seaborn": _parse_log_seaborn,
    "pallets/flask": _parse_log_pytest,
    "psf/requests": _parse_log_pytest_options,
    "pvlib/pvlib-python": _parse_log_pytest,
    "pydata/xarray": _parse_log_pytest,
    "pydicom/pydicom": _parse_log_pytest_options,
    "pylint-dev/astroid": _parse_log_pytest,
    "pylint-dev/pylint": _parse_log_pytest_options,
    "pytest-dev/pytest": _parse_log_pytest,
    "pyvista/pyvista": _parse_log_pytest,
    "scikit-learn/scikit-learn": _parse_log_pytest_v2,
    "sqlfluff/sqlfluff": _parse_log_pytest,
    "sphinx-doc/sphinx": _parse_log_pytest_v2,
    "sympy/sympy": _parse_log_sympy,
    # Go
    "caddyserver/caddy": _parse_log_gotest,
    "hashicorp/terraform": _parse_log_gotest,
    "prometheus/prometheus": _parse_log_gotest,
    "gohugoio/hugo": _parse_log_gotest,
    "gin-gonic/gin": _parse_log_gotest,
    # Java
    "google/gson": _parse_log_maven,
    "apache/druid": _parse_log_maven,
    "javaparser/javaparser": _parse_log_maven,
    "projectlombok/lombok": _parse_log_ant,
    "apache/lucene": _parse_log_gradle,
    "reactivex/rxjava": _parse_log_gradle,
    # PHP
    "phpoffice/phpspreadsheet": _parse_log_phpunit,
    "laravel/framework": _parse_log_phpunit,
    "php-cs-fixer/php-cs-fixer": _parse_log_phpunit,
    "briannesbitt/carbon": _parse_log_phpunit,
    # Ruby
    "fluent/fluentd": _parse_log_ruby_unit,
    "fastlane/fastlane": _parse_log_rspec_json,
    "jordansissel/fpm": _parse_log_rspec_json,
    "faker-ruby/faker": _parse_log_ruby_unit,
    "rubocop/rubocop": _parse_log_rspec_json,
    # Rust
    "burntsushi/ripgrep": _parse_log_cargo,
    "sharkdp/bat": _parse_log_cargo,
    "astral-sh/ruff": _parse_log_cargo,
    "tokio-rs/tokio": _parse_log_cargo,
    "uutils/coreutils": _parse_log_cargo,
    "nushell/nushell": _parse_log_cargo,
    "tokio-rs/axum": _parse_log_cargo,
    # C
    "redis/redis": _parse_log_redis,
    "jqlang/jq": _parse_log_jq,
    "nlohmann/json": _parse_log_doctest,
    "micropython/micropython": _parse_log_micropython,
    "valkey-io/valkey": _parse_log_redis,
    "fmtlib/fmt": _parse_log_googletest,
    # JS
    "babel/babel": _parse_log_jest,
    "vuejs/core": _parse_log_vitest,
    "facebook/docusaurus": _parse_log_jest,
    "mrdoob/three.js": _parse_log_tap,
    "axios/axios": _parse_log_tap,
    "markedjs/marked": _parse_log_marked,
    "diegomura/react-pdf": _parse_log_react_pdf,
}


def _get_parser(parser_name: str, instance_id: str) -> object | None:
    """Resolve a log parser, handling jekyll/immutable-js special cases."""
    if parser_name == "parse_log_jekyll" or parser_name == "jekyll/jekyll":
        return lambda log: _parse_log_jekyll(log, instance_id)
    if parser_name in ("parse_log_immutable_js", "immutable-js/immutable-js"):
        pr = instance_id.split("-")[-1]
        return _parse_log_jest_json if pr == "2005" else _parse_log_jest
    # parse_log_calypso, parse_log_p5js, parse_log_karma, parse_log_chart_js: no parser key clash
    return _LOG_PARSERS.get(parser_name)


# ---------------------------------------------------------------------------
# Patch/test file utilities (mirrors SWE-rebench fork utils)
# ---------------------------------------------------------------------------

def _get_modified_files(patch: str) -> list[str]:
    """Files modified (not newly created) – used to reset test files."""
    result = []
    for line in patch.split("\n"):
        if line.startswith("--- a/"):
            result.append(line[6:])
    return result


def _get_changed_files(patch: str) -> list[str]:
    """All files added or modified – used for test directives."""
    result = []
    for line in patch.split("\n"):
        if line.startswith("+++ b/"):
            path = line[6:]
            if path and path != "/dev/null":
                result.append(path)
    return result


def _get_test_directives(instance: dict) -> list[str]:
    """
    Get test directives from test_patch (mirrors get_test_directives in SWE-rebench fork).
    Handles Django path transformation.
    """
    repo = instance.get("repo", "")
    test_patch = instance.get("test_patch", "")

    if repo in ("swe-bench/humaneval", "swe-bench/humanevalfix-python"):
        return ["test.py"]
    if repo in ("swe-bench/humanevalfix-go", "swe-bench/humanevalfix-java"):
        return []
    if repo == "swe-bench/humanevalfix-js":
        return ["test.js"]
    if repo == "nebius/nebo":
        return [""]

    directives = [d for d in _get_changed_files(test_patch)
                  if not any(d.endswith(ext) for ext in _NON_TEST_EXTS)]

    if repo == "django/django":
        transformed = []
        for d in directives:
            d = d[:-3] if d.endswith(".py") else d
            d = d[6:] if d.startswith("tests/") else d
            d = d.replace("/", ".")
            transformed.append(d)
        directives = transformed

    return directives


def _normalize_test_name(name: str) -> str:
    for pattern in _TIMING_NORMALIZE_RES:
        name = pattern.sub("", name)
    return name.strip()


def _b64write_cmd(path: str, content: str) -> str:
    b64 = base64.b64encode(content.encode()).decode()
    return f"python3 -c \"import base64; open({path!r},'wb').write(base64.b64decode({b64!r}))\""


# Max command length for a single _b64write_cmd (~64KB raw → ~90KB base64 → safe single command)
_B64WRITE_CHUNK_SIZE = 48000  # bytes of raw content per chunk


def _upload_file(env, path: str, content: str, *, timeout: int = 30) -> None:
    """Upload content to a file in the environment, chunking if necessary."""
    raw = content.encode()
    if len(raw) <= _B64WRITE_CHUNK_SIZE:
        env.execute({"command": _b64write_cmd(path, content)}, timeout=timeout)
    else:
        # Write in chunks using append mode
        env.execute({"command": f"rm -f {path}"}, timeout=10)
        for offset in range(0, len(raw), _B64WRITE_CHUNK_SIZE):
            chunk = raw[offset:offset + _B64WRITE_CHUNK_SIZE]
            b64 = base64.b64encode(chunk).decode()
            cmd = f"python3 -c \"import base64; open({path!r},'ab').write(base64.b64decode({b64!r}))\""
            env.execute({"command": cmd}, timeout=timeout)


# ---------------------------------------------------------------------------
# Eval script building (mirrors make_eval_script_list_py / _common)
# ---------------------------------------------------------------------------

def _is_python_instance(install_config: dict) -> bool:
    """Heuristic: Python repos use conda (have packages/python key in install_config)."""
    return bool(install_config.get("packages") or install_config.get("python"))


def _command_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(command) for command in value if str(command).strip()]


def build_environment_setup_script(instance: dict) -> str:
    """Build the dependency/setup phase used by the official evaluator.

    Persistent verifier sandboxes run this once, then reuse the prepared environment
    while individual eval scripts reset repository state and skip repeated installs.
    """
    install_config = instance.get("install_config", {}) or {}
    lines = ["#!/bin/bash", "set -eo pipefail"]
    if _is_python_instance(install_config):
        lines += [
            "source /opt/miniconda3/bin/activate",
            "conda activate testbed",
            f"cd {DOCKER_WORKDIR}",
            *_command_list(install_config.get("eval_commands")),
            *_command_list(install_config.get("install")),
        ]
    return "\n".join(lines)


def prepare_environment_for_evaluation(env, instance: dict, *, timeout: int = 600) -> dict:
    """Prepare a caller-owned verifier sandbox once with official setup commands."""
    script = build_environment_setup_script(instance)
    _upload_file(env, "/tmp/official-prepare.sh", script, timeout=30)
    started = time.perf_counter()
    result = env.execute({"command": "bash /tmp/official-prepare.sh 2>&1"}, timeout=timeout)
    duration = time.perf_counter() - started
    output = str(result.get("output") or "") + str(result.get("exception_info") or "")
    if result.get("returncode") != 0:
        raise RuntimeError(
            "official verifier environment setup failed "
            f"(rc={result.get('returncode')}): {output[-2000:]}"
        )
    return {
        "returncode": result.get("returncode"),
        "duration_sec": duration,
        "output_tail": output[-2000:],
    }


def build_eval_script(
    instance: dict,
    *,
    test_command_override: str | None = None,
    environment_prepared: bool = False,
) -> str:
    """
    Build the eval script following SWE-rebench fork's make_eval_script_list_py
    (Python repos) or make_eval_script_list_common (other languages).
    The model patch is NOT applied here – it is applied separately before running
    this script (mirrors how run_evaluation.py copies patch.diff then runs eval.sh).
    The test_patch is uploaded separately as /tmp/test_patch.diff to avoid
    embedding large patches in the eval script via heredoc.

    A persistent verifier passes ``environment_prepared=True`` only after its
    one-time official setup succeeds. Non-Python build commands remain part of
    every grade because they must see the submitted source patch.
    """
    install_config = instance.get("install_config", {})
    base_commit = instance.get("_persistent_baseline_commit") or instance.get(
        "base_commit", "HEAD"
    )
    test_patch = instance.get("test_patch", "")
    repo_dir = DOCKER_WORKDIR
    env_name = "testbed"

    test_files = _get_modified_files(test_patch)
    reset_cmd = f"git checkout {base_commit} {' '.join(test_files)}" if test_files else 'echo "No test files to reset"'
    apply_test_patch = "git apply -v /tmp/test_patch.diff"

    if _is_python_instance(install_config):
        # Python path (make_eval_script_list_py)
        test_cmd = install_config.get("test_cmd", "")
        if isinstance(test_cmd, list):
            test_cmd = " ".join(test_cmd)
        directives = _get_test_directives(instance)
        test_command = (
            test_command_override
            if test_command_override is not None
            else " ".join([test_cmd, *directives]).strip()
        )

        lines = [
            "#!/bin/bash",
            "set -o pipefail",
            "source /opt/miniconda3/bin/activate",
            f"conda activate {env_name}",
            f"cd {repo_dir}",
        ]
        if not environment_prepared:
            lines += _command_list(install_config.get("eval_commands"))
        lines += [
            f"git config --global --add safe.directory {repo_dir}",
            f"cd {repo_dir}",
            "git status",
            "git show",
            f"git -c core.fileMode=false diff {base_commit}",
            "source /opt/miniconda3/bin/activate",
            f"conda activate {env_name}",
        ]
        if not environment_prepared:
            lines += _command_list(install_config.get("install"))
        lines += [
            reset_cmd,
            apply_test_patch,
            f"echo '{START_TEST_OUTPUT}'",
            test_command,
            f"echo '{END_TEST_OUTPUT}'",
            reset_cmd,
        ]
    else:
        # Common path (make_eval_script_list_common)
        if test_command_override is not None:
            test_cmds = [test_command_override]
        else:
            test_cmds = install_config.get("test_cmd", [])
            if isinstance(test_cmds, str):
                test_cmds = [test_cmds]

        build_cmds = _command_list(install_config.get("build"))

        lines = [
            "#!/bin/bash",
            "set -o pipefail",
            f"cd {repo_dir}",
            f"git config --global --add safe.directory {repo_dir}",
            f"cd {repo_dir}",
            reset_cmd,
            apply_test_patch,
            *build_cmds,
            f"echo '{START_TEST_OUTPUT}'",
            *test_cmds,
            f"echo '{END_TEST_OUTPUT}'",
            reset_cmd,
        ]

    return "\n".join(lines)


def grade_eval_output(
    instance: dict,
    output: str,
    *,
    exit_code: int,
    patch_applied: bool,
) -> dict:
    """Grade one SWE-rebench eval log independently of sandbox ownership."""
    instance_id = instance["instance_id"]
    install_config = instance.get("install_config", {})
    repo = instance.get("repo", "")
    parser_name = install_config.get("log_parser") or repo
    fail_to_pass = [_normalize_test_name(n) for n in instance.get("FAIL_TO_PASS", [])]
    pass_to_pass = [_normalize_test_name(n) for n in instance.get("PASS_TO_PASS", [])]
    fail_only = repo in _FAIL_ONLY_REPOS

    bad_codes = [c for c in [APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT] if c in output]
    if bad_codes or (START_TEST_OUTPUT not in output or END_TEST_OUTPUT not in output):
        if bad_codes:
            logger.warning(f"{instance_id}: bad signals in output: {bad_codes}")
        else:
            logger.warning(f"{instance_id}: START/END test markers not found in output")
        parsed: dict[str, str] = {}
    else:
        parsed = _parse_test_output(output, parser_name, instance_id)
        parsed = {_normalize_test_name(k): v for k, v in parsed.items()}

    passed = {k for k, v in parsed.items() if v in ("PASSED", "XFAIL")}

    def _check(case: str) -> bool:
        if fail_only:
            return case not in parsed or parsed.get(case) != "FAILED"
        return case in parsed and parsed[case] in ("PASSED", "XFAIL")

    f2p_passed = sorted(t for t in fail_to_pass if _check(t))
    f2p_failed = sorted(t for t in fail_to_pass if not _check(t))
    p2p_passed = sorted(passed & set(pass_to_pass))
    p2p_failed = sorted(set(pass_to_pass) - passed)
    resolved = len(fail_to_pass) > 0 and not f2p_failed and (fail_only or not p2p_failed)

    return {
        "instance_id": instance_id,
        "resolved": resolved,
        "exit_code": exit_code,
        "patch_applied": patch_applied,
        "fail_to_pass_passed": f2p_passed,
        "fail_to_pass_failed": f2p_failed,
        "pass_to_pass_passed": len(p2p_passed),
        "pass_to_pass_failed": p2p_failed,
        "parsed_tests_count": len(parsed),
        "error": "",
    }


def _parse_test_output(log: str, parser_name: str, instance_id: str) -> dict[str, str]:
    """Extract START/END section and run the appropriate log parser."""
    if START_TEST_OUTPUT in log and END_TEST_OUTPUT in log:
        log = log.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
    parser = _get_parser(parser_name, instance_id)
    if parser is None:
        logger.warning(f"Unknown log_parser {parser_name!r} for {instance_id}, falling back to parse_log_pytest_v2")
        parser = _parse_log_pytest_v2
    return parser(log)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def create_environment(instance: dict, env_config: dict):
    """Create, but do not evaluate or clean up, one official verifier sandbox."""
    from minisweagent.environments.extra.azure_modal import AzureModalEnvironment

    image = instance["image_name"]
    if image.startswith("docker.io/"):
        image = image.replace("docker.io/", "mirror.gcr.io/")  # redirect docker.io to mirror.gcr.io
    elif not image.startswith("mirror.gcr.io/"):
        image = "mirror.gcr.io/" + image  # ensure all images are pulled from mirror.gcr.io for better performance
    cfg = {k: v for k, v in env_config.items() if k not in ("environment_class",)}
    cfg.update(image=image, cwd=DOCKER_WORKDIR)
    return AzureModalEnvironment(**cfg)


def evaluate_instance_in_environment(
    instance: dict,
    patch: str,
    env,
    env_config: dict,
    *,
    logs_dir: Path | None = None,
    sample_idx: int = 0,
    environment_prepared: bool = False,
) -> dict:
    """Evaluate in a caller-owned sandbox, resetting it to baseline first."""
    instance_id = instance["instance_id"]
    base_commit = instance.get("_persistent_baseline_commit")
    if base_commit:
        reset = env.execute(
            {
                "command": (
                    f"cd {DOCKER_WORKDIR} && git reset --hard {shlex.quote(str(base_commit))} "
                    "&& git clean -fdq"
                )
            },
            timeout=120,
        )
        if reset["returncode"] != 0:
            result = grade_eval_output(
                instance, "", exit_code=reset["returncode"], patch_applied=False
            )
            result["error"] = (
                f"baseline reset failed: {str(reset.get('output') or '')[-1000:]}"
            )
            return result

    eval_script = build_eval_script(instance, environment_prepared=environment_prepared)
    test_timeout = env_config.get("timeout", env_config.get("sandbox_timeout", 600))
    _upload_file(env, "/tmp/patch.diff", patch or "", timeout=30)
    _upload_file(env, "/tmp/test_patch.diff", instance.get("test_patch", ""), timeout=60)
    _upload_file(env, "/tmp/eval.sh", eval_script, timeout=30)

    patch_applied = not bool((patch or "").strip())
    if not patch_applied:
        for git_apply_cmd in [
            "git apply --verbose /tmp/patch.diff",
            "git apply --verbose --reject /tmp/patch.diff",
            "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
        ]:
            res = env.execute({"command": f"cd {DOCKER_WORKDIR} && {git_apply_cmd}"}, timeout=60)
            if res["returncode"] == 0:
                patch_applied = True
                break
    if not patch_applied:
        logger.warning("%s: model patch failed to apply", instance_id)

    run = env.execute({"command": "bash /tmp/eval.sh"}, timeout=test_timeout)
    output = run["output"]

    if logs_dir:
        logs_dir.mkdir(parents=True, exist_ok=True)
        suffix = f".sample_{sample_idx}" if sample_idx > 0 else ""
        (logs_dir / f"{instance_id}{suffix}_log.txt").write_text(output)

    return grade_eval_output(
        instance,
        output,
        exit_code=run["returncode"],
        patch_applied=patch_applied,
    )


def evaluate_instance_azure_modal(
    instance: dict,
    patch: str,
    env_config: dict,
    *,
    logs_dir: Path | None = None,
    sample_idx: int = 0,
) -> dict:
    """Evaluate a single instance inside a fresh Azure Modal sandbox."""
    env = create_environment(instance, env_config)
    try:
        prepare_environment_for_evaluation(
            env,
            instance,
            timeout=int(env_config.get("sandbox_timeout", 600)),
        )
        return evaluate_instance_in_environment(
            instance,
            patch,
            env,
            env_config,
            logs_dir=logs_dir,
            sample_idx=sample_idx,
            environment_prepared=True,
        )
    finally:
        env.cleanup()


# ---------------------------------------------------------------------------
# Trajectory scanning and I/O helpers
# ---------------------------------------------------------------------------

def load_patch_from_traj(traj_path: Path) -> str:
    data = json.loads(traj_path.read_text())
    # Native format: info.submission
    patch = data.get("info", {}).get("submission", "")
    if patch:
        return patch
    # OpenHands format: metadata.model_patch
    return data.get("metadata", {}).get("model_patch", "") or ""


def _parse_traj_filename(name: str, suffix: str) -> tuple[str, int]:
    """Extract (instance_id, sample_idx) from a native trajectory filename.

    Expected format: ``instance_id.sample_N.traj`` (suffix=``.traj``).
    """
    name = name.removesuffix(suffix)
    if ".sample_" in name:
        instance_id, _, idx = name.rpartition(".sample_")
        return instance_id, int(idx)
    return name, 0


def scan_trajectories(output_dir: Path, traj_format: str = "auto") -> dict[tuple[str, int], Path]:
    """Return {(instance_id, sample_idx): traj_path} for all trajectory files in output_dir.

    Args:
        output_dir: Directory to scan for trajectory files.
        traj_format: ``"native"`` for ``*.traj.json``, ``"openhands"`` for ``*.jsonl``,
            or ``"auto"`` (default) to scan both.
    """
    trajs: dict[tuple[str, int], Path] = {}
    if traj_format in ("auto", "native"):
        for traj_file in sorted(output_dir.rglob("*.traj.json")):
            instance_id, sample_idx = _parse_traj_filename(traj_file.stem, ".traj")
            trajs[(instance_id, sample_idx)] = traj_file
    if traj_format in ("auto", "openhands"):
        for traj_file in sorted(output_dir.rglob("*.jsonl")):
            try:
                data = json.loads(traj_file.read_text())
                instance_id = data["instance_id"]
                sample_idx = int(data.get("sample_idx", 0))
            except (json.JSONDecodeError, KeyError, ValueError):
                logger.warning(f"Skipping unrecognised .jsonl file: {traj_file}")
                continue
            trajs[(instance_id, sample_idx)] = traj_file
    return trajs


def get_existing_verified_samples(results_path: Path) -> set[tuple[str, int]]:
    """Return set of (instance_id, sample_idx) pairs already verified."""
    if not results_path.exists():
        return set()
    data = json.loads(results_path.read_text())
    existing = set()
    for instance_id, entry in data.items():
        for s in entry.get("samples", []):
            existing.add((instance_id, s["sample_idx"]))
        if "samples" not in entry:
            existing.add((instance_id, 0))
    return existing


def update_results_file(output_path: Path, instance_id: str, sample_idx: int, result: dict):
    global _RESULTS_CACHE, _RESULTS_DIRTY
    with _OUTPUT_FILE_LOCK:
        if _RESULTS_CACHE is None:
            _RESULTS_CACHE = json.loads(output_path.read_text()) if output_path.exists() and output_path.stat().st_size > 0 else {}
        entry = _RESULTS_CACHE.get(instance_id, {"instance_id": instance_id, "samples": []})
        samples = [s for s in entry.get("samples", []) if s.get("sample_idx") != sample_idx]
        result["sample_idx"] = sample_idx
        samples.append(result)
        samples.sort(key=lambda s: s["sample_idx"])
        entry["samples"] = samples
        entry["resolved"] = any(s.get("resolved") for s in samples)
        _RESULTS_CACHE[instance_id] = entry
        _RESULTS_DIRTY = True


def flush_results_file(output_path: Path):
    global _RESULTS_DIRTY
    with _OUTPUT_FILE_LOCK:
        if _RESULTS_CACHE is not None and _RESULTS_DIRTY:
            output_path.write_text(json.dumps(_RESULTS_CACHE, indent=2))
            _RESULTS_DIRTY = False


def process_instance(
    instance: dict,
    traj_path: Path | None,
    results_path: Path,
    env_config: dict,
    logs_dir: Path,
    progress_manager: RunBatchProgressManager,
    sample_idx: int = 0,
    golden_patch: str | None = None,
) -> None:
    instance_id = instance["instance_id"]
    task_label = f"{instance_id}#{sample_idx}" if sample_idx > 0 else instance_id
    progress_manager.on_instance_start(task_label)
    progress_manager.update_instance_status(task_label, "Starting sandbox")
    result: dict = {}
    exit_status = "error"
    try:
        patch = golden_patch if golden_patch is not None else load_patch_from_traj(traj_path)
        progress_manager.update_instance_status(task_label, "Running tests")
        result = evaluate_instance_azure_modal(instance, patch, env_config, logs_dir=logs_dir, sample_idx=sample_idx)
        exit_status = "resolved" if result["resolved"] else "failed"
    except Exception as e:
        logger.error(f"Error verifying {task_label}: {e}", exc_info=True)
        result = {
            "instance_id": instance_id,
            "resolved": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "fail_to_pass_passed": [],
            "fail_to_pass_failed": [],
            "pass_to_pass_passed": 0,
            "pass_to_pass_failed": [],
        }
    progress_manager.on_instance_end(task_label, exit_status)
    update_results_file(results_path, instance_id, sample_idx, result)


def _print_verification_report(results: dict, results_path: Path) -> None:
    """Print verification report from results dictionary."""
    total_instances = len(results)
    resolved_instances = sum(1 for r in results.values() if r.get("resolved"))
    total_samples = sum(len(r.get("samples", [])) for r in results.values())
    resolved_samples = sum(
        sum(1 for s in r.get("samples", []) if s.get("resolved"))
        for r in results.values()
    )

    console.print(f"\n[bold]{'=' * 80}[/bold]")
    console.print(f"[bold cyan]VERIFICATION REPORT[/bold cyan]".center(80))
    console.print(f"[bold]{'=' * 80}[/bold]")
    console.print(f"\nResults file: [bold green]{results_path}[/bold green]")
    console.print(f"\n[bold]Pass@k (instance-level):[/bold]")
    console.print(f"  Resolved: {resolved_instances}/{total_instances} ({resolved_instances / max(total_instances, 1) * 100:.1f}%)")
    console.print(f"\n[bold]Per-sample:[/bold]")
    console.print(f"  Resolved: {resolved_samples}/{total_samples} ({resolved_samples / max(total_samples, 1) * 100:.1f}%)")

    # Group by resolution status
    resolved_ids = [iid for iid, r in results.items() if r.get("resolved")]
    failed_ids = [iid for iid, r in results.items() if not r.get("resolved")]

    if resolved_ids:
        console.print(f"\n[bold green]Resolved instances ({len(resolved_ids)}):[/bold green]")
        for iid in sorted(resolved_ids)[:20]:
            console.print(f"  ✓ {iid}")
        if len(resolved_ids) > 20:
            console.print(f"  ... and {len(resolved_ids) - 20} more")

    if failed_ids:
        console.print(f"\n[bold red]Failed instances ({len(failed_ids)}):[/bold red]")
        for iid in sorted(failed_ids)[:20]:
            console.print(f"  ✗ {iid}")
        if len(failed_ids) > 20:
            console.print(f"  ... and {len(failed_ids) - 20} more")

    console.print(f"\n[bold]{'=' * 80}[/bold]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# fmt: off
@app.command()
def main(
    subset: str = typer.Option("rebench_v2", "--subset", help="SWE-rebench subset or dataset path", rich_help_panel="Data selection"),
    split: str = typer.Option("train", "--split", help="Dataset split", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5')", rich_help_panel="Data selection"),
    output: str = typer.Option(..., "-o", "--output", help="Directory containing trajectory files to verify", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of parallel workers", rich_help_panel="Basic"),
    config_spec: list[str] = typer.Option([str(DEFAULT_CONFIG_FILE)], "-c", "--config", help="Config files/specs for azure_modal environment", rich_help_panel="Basic"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Re-verify instances already in results file", rich_help_panel="Data selection"),
    logs_dir: str = typer.Option("", "--logs-dir", help="Directory to save test logs (default: <output>/verify_logs)", rich_help_panel="Advanced"),
    verify_output: str = typer.Option("", "--verify-output", help="Output results file (default: <output>/verify_results_azure_modal.json)", rich_help_panel="Basic"),
    golden: bool = typer.Option(False, "--golden", help="Sanity check: evaluate golden patches from the dataset instead of model patches", rich_help_panel="Advanced"),
    report_only: bool = typer.Option(False, "--report-only", help="Only print report from existing results file without running verification", rich_help_panel="Advanced"),
    traj_format: str = typer.Option("auto", "--traj-format", help="Trajectory format: 'native' (*.traj.json), 'openhands' (*.jsonl), or 'auto' (both)", rich_help_panel="Data selection"),
) -> None:
    # fmt: on
    """Verify patches from trajectory files using Azure Modal sandboxes."""
    output_path = Path(output)
    if golden:
        default_results = output_path / "verify_results_golden.json"
        default_logs = output_path / "verify_logs_golden"
    else:
        default_results = output_path / "verify_results_azure_modal.json"
        default_logs = output_path / "verify_logs"
    results_path = Path(verify_output) if verify_output else default_results
    logs_path = Path(logs_dir) if logs_dir else default_logs

    # Report-only mode: just print statistics from existing results file
    if report_only:
        if not results_path.exists():
            console.print(f"[red]Results file not found: {results_path}[/red]")
            raise typer.Exit(1)
        results = json.loads(results_path.read_text())
        if filter_spec or slice_spec:
            ids = sorted(results.keys())
            if filter_spec:
                ids = [iid for iid in ids if re.match(filter_spec, iid)]
            if slice_spec:
                values = [int(x) if x else None for x in slice_spec.split(":")]
                ids = ids[slice(*values)]
            results = {iid: results[iid] for iid in ids}
        _print_verification_report(results, results_path)
        return

    add_file_handler(output_path / "minisweagent_verify.log")

    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = {inst["instance_id"]: inst for inst in load_dataset(dataset_path, split=split)}

    if golden:
        # Golden mode: evaluate dataset golden patches, no trajectory files needed
        instance_ids = sorted(instances.keys())
    else:
        logger.info(f"Scanning trajectories in {output_path}")
        traj_map = scan_trajectories(output_path, traj_format=traj_format)
        if not traj_map:
            console.print(f"[yellow]No trajectory files found in {output_path}[/yellow]")
            raise typer.Exit(1)
        logger.info(f"Found {len(traj_map)} trajectory files")

        traj_instance_ids = sorted({iid for iid, _ in traj_map})
        instance_ids = [iid for iid in traj_instance_ids if iid in instances]
        missing = [iid for iid in traj_instance_ids if iid not in instances]
        if missing:
            logger.warning(f"{len(missing)} trajectory instance_ids not found in dataset: {missing[:5]}...")

    if filter_spec:
        instance_ids = [iid for iid in instance_ids if re.match(filter_spec, iid)]
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instance_ids = instance_ids[slice(*values)]

    if golden:
        # Golden mode: one work item per instance (sample_idx=0, no traj_path)
        work_items_golden = [(iid, 0) for iid in instance_ids]
        if not redo_existing:
            existing = get_existing_verified_samples(results_path)
            before = len(work_items_golden)
            work_items_golden = [(iid, sidx) for iid, sidx in work_items_golden if (iid, sidx) not in existing]
            if before > len(work_items_golden):
                logger.info(f"Skipping {before - len(work_items_golden)} already-verified instances")
        logger.info(f"[golden] Verifying {len(work_items_golden)} instances with golden patches...")
    else:
        # Normal mode: all (instance_id, sample_idx) pairs from trajectories
        instance_id_set = set(instance_ids)
        work_items = [(iid, sidx, traj_map[(iid, sidx)]) for iid, sidx in sorted(traj_map) if iid in instance_id_set]
        if not redo_existing:
            existing = get_existing_verified_samples(results_path)
            before = len(work_items)
            work_items = [(iid, sidx, tp) for iid, sidx, tp in work_items if (iid, sidx) not in existing]
            if before > len(work_items):
                logger.info(f"Skipping {before - len(work_items)} already-verified samples")
        logger.info(f"Verifying {len(work_items)} samples across {len({iid for iid, _, _ in work_items})} instances...")

    configs = [get_config_from_spec(spec) for spec in config_spec]
    config = recursive_merge(*configs)
    env_config = config.get("environment", {})

    n_jobs = len(work_items_golden) if golden else len(work_items)
    progress_manager = RunBatchProgressManager(
        n_jobs, output_path / f"verify_statuses_{time.time()}.yaml"
    )

    if workers <= 1:
        live_ctx = Live(progress_manager.render_group, refresh_per_second=4)
    else:
        progress_manager._main_progress_bar.live.redirect_stdout = False
        progress_manager._main_progress_bar.live.redirect_stderr = False
        live_ctx = progress_manager._main_progress_bar

    with live_ctx:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        futures = {}
        if golden:
            for iid, s_idx in work_items_golden:
                future = executor.submit(
                    process_instance,
                    instances[iid], None, results_path, env_config, logs_path, progress_manager, s_idx,
                    golden_patch=instances[iid].get("patch", ""),
                )
                futures[future] = iid
        else:
            for iid, s_idx, traj_path in work_items:
                task_label = f"{iid}#{s_idx}" if s_idx > 0 else iid
                future = executor.submit(
                    process_instance,
                    instances[iid], traj_path, results_path, env_config, logs_path, progress_manager, s_idx,
                )
                futures[future] = task_label
        last_flush = time.time()
        try:
            done: set = set()
            while len(done) < len(futures):
                try:
                    for future in concurrent.futures.as_completed(futures, timeout=2.0):
                        done.add(future)
                        try:
                            future.result()
                        except concurrent.futures.CancelledError:
                            pass
                        except Exception as e:
                            task_label = futures[future]
                            logger.error(f"Uncaught error for {task_label}: {e}", exc_info=True)
                            progress_manager.on_uncaught_exception(task_label, e)
                except concurrent.futures.TimeoutError:
                    pass
                if time.time() - last_flush > 30:
                    flush_results_file(results_path)
                    last_flush = time.time()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Cancelling pending tasks and waiting for in-flight tasks to finish...")
            console.print("\n[yellow]Interrupted! Waiting for in-flight tasks to save results (up to 30s)...[/yellow]")
            executor.shutdown(wait=False, cancel_futures=True)
            in_flight = [f for f in futures if not f.done()]
            if in_flight:
                concurrent.futures.wait(in_flight, timeout=30)
            flush_results_file(results_path)
            console.print(f"[green]Partial results saved to {results_path}[/green]")
            console.print("[green]Re-run the same command to resume from where you left off.[/green]")
            raise SystemExit(1)
        finally:
            executor.shutdown(wait=False)

    flush_results_file(results_path)
    if _RESULTS_CACHE:
        _print_verification_report(_RESULTS_CACHE, results_path)


if __name__ == "__main__":
    app()
