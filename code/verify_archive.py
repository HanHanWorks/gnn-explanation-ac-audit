#!/usr/bin/env python3
"""Verify the RESS revision archive without running the experiments.

Checks:
  1. the 137 formal artefacts indexed in manifest.json are present;
  2. every Python file compiles under the current interpreter;
  3. the mapping tables cover all benchmark branches;
  4. the figure CSV files exist and have the expected columns;
  5. the archive checksum manifest is present and well formed.
"""
import csv, hashlib, json, os, py_compile, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ok = True


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    ok = ok and bool(cond)


manifest = json.load(open(os.path.join(ROOT, "manifest.json")))
formal = [m["source"] for m in manifest]
def formal_path(f):
    if f.startswith("../phase0/ckpt/"):
        f = "phase0_ckpt/" + f[len("../phase0/ckpt/"):]
    return os.path.join(ROOT, "data", "results_json", f)
present = [f for f in formal if os.path.exists(formal_path(f))]
check(len(formal) == 137, f"manifest lists 137 formal artefacts (found {len(formal)})")
check(len(present) == 137, f"all 137 formal artefacts present (found {len(present)})")

bad = []
for dirpath, _, files in os.walk(os.path.join(ROOT, "code")):
    for f in files:
        if f.endswith(".py"):
            try:
                py_compile.compile(os.path.join(dirpath, f), doraise=True)
            except Exception as exc:
                bad.append((f, str(exc).splitlines()[-1]))
check(not bad, f"all Python files compile ({len(bad)} failures)")
for f, e in bad:
    print("  ", f, e)

for system, n in [("ieee24", 38), ("ieee39", 46), ("uk", 99)]:
    p = os.path.join(ROOT, "mapping", f"{system}_branch_map.csv")
    rows = list(csv.reader(open(p)))
    check(len(rows) - 1 == n, f"{system} mapping has {n} branches (found {len(rows)-1})")

for name, cols in [
    ("figure_2_outcomes.csv", ["system", "method", "pfr"]),
    ("figure_3_quality.csv", ["system", "ranking", "bacc", "pfr"]),
    ("figure_3_null.csv", ["system", "seed", "bacc"]),
    ("figure_4_transfer.csv", ["system", "seed", "contingency_bacc"]),
    ("figure_5_protocol.csv", ["panel", "system", "item", "value"]),
]:
    p = os.path.join(ROOT, "data", "figure_csv", name)
    if not os.path.exists(p):
        check(False, f"{name} exists")
        continue
    hdr = next(csv.reader(open(p)))
    check(all(c in hdr for c in cols), f"{name} has required columns")

mf = os.path.join(ROOT, "MANIFEST.sha256")
check(os.path.exists(mf), "MANIFEST.sha256 exists")
if os.path.exists(mf):
    lines = [l for l in open(mf).read().splitlines() if l.strip()]
    check(len(lines) >= 400, f"checksum manifest has {len(lines)} entries")
    bad_hash = []
    for line in lines:
        digest, rel = line.split("  ", 1)
        p = os.path.join(ROOT, rel.lstrip("./"))
        if not os.path.exists(p) or sha256(p) != digest:
            bad_hash.append(rel)
    check(not bad_hash, f"all {len(lines)} checksums match")
    for rel in bad_hash[:5]:
        print("   mismatch:", rel)

den = os.path.join(ROOT, "cascade_denominators.json")
if os.path.exists(den):
    d = json.load(open(den))
    check(all(k in d for k in ("ieee24", "ieee39", "uk")),
          "population denominators present for all three systems")
else:
    check(False, "cascade_denominators.json exists")

print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
