"""Fetch, patch, and build DREAMPlace for the hrt_winner submission.

This is the **replication script** for setting up the DREAMPlace dependency
needed by `dpl_wrapper.py`. Run it once before invoking the placer:

    python submissions/hrt_winner/setup_dreamplace.py

It will:

1. Fetch DREAMPlace-master + 5 submodules from a GitHub tarball mirror.
   - Primary source: github.com/limbo018/DREAMPlace
   - On networks without direct GitHub access (e.g. internal CI), an
     Artifactory mirror in the same path layout works:
         https://<artifactory>/artifactory/shared-github-remote/<user>/<repo>/archive/refs/heads/<branch>.tar.gz
     Set the `DPL_SOURCE_BASE` env var to point at it.
2. Apply small source patches needed to build with modern CMake (>=4.x) and
   NumPy 2.x (np.string_ removed).
3. Configure + build with the project's .venv Python and conda Boost/Bison/
   Flex/zlib. CPU-only build; the GPU operators are skipped because they
   require nvcc, but the FFT/placement core works fine on CPU.
4. Copy DREAMPlace's Python sources into the build/dreamplace tree, plus
   the params.json schema, so dpl_wrapper can invoke Placer.py.
5. Verify the build by importing dreamplace.ops.place_io.

The destination is fixed at /tmp/dreamplace_src/DREAMPlace-master. If you
re-run, it will skip steps that already succeeded. Pass --force to redo.

Build requirements on the host:
    - Python 3.10 venv at /home/coder/macro-place-challenge-2026/.venv with
      torch (CPU build is fine), numpy, shapely, cairocffi, torch_optimizer,
      ncg-optimizer installed.
    - conda env "base" with: cmake>=3.20, bison, flex, zlib, boost-cpp.
      The build picks these up via /opt/conda/{bin,include,lib}.
    - gcc/g++ from the system (tested with gcc 14).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# --- Configuration ----------------------------------------------------------

DPL_BRANCH = "master"
SUBMODULES: list[tuple[str, str, str]] = [
    # (path-under-thirdparty, github-user/repo, branch)
    ("Limbo",          "limbo018/Limbo",           "master"),
    ("munkres-cpp",    "saebyn/munkres-cpp",       "master"),
    ("cub",            "NVlabs/cub",               "1.8.0"),
    ("pybind11",       "pybind/pybind11",          "master"),
    ("OpenTimer",      "OpenTimer/OpenTimer",      "v2.0.0"),
]
# Limbo has its own sub-submodule:
SUB_SUBMODULES: list[tuple[str, str, str, str]] = [
    # (limbo-relative-path, github-user/repo, branch, target-name)
    ("thirdparty/OpenBLAS", "OpenMathLib/OpenBLAS", "develop", "OpenBLAS"),
]

DEFAULT_SOURCE_BASE = os.environ.get(
    "DPL_SOURCE_BASE",
    "https://github.com/{user}/{repo}/archive/refs/heads/{branch}.tar.gz",
)

DPL_ROOT = Path("/tmp/dreamplace_src/DREAMPlace-master")
DPL_BUILD = DPL_ROOT / "build"
QM_STUB = Path("/tmp/qm_stub")
VENV_PY = Path("/home/coder/macro-place-challenge-2026/.venv/bin/python")


# --- Fetch helpers ----------------------------------------------------------


def _tarball_url(user_repo: str, branch: str) -> str:
    user, repo = user_repo.split("/", 1)
    return DEFAULT_SOURCE_BASE.format(user=user, repo=repo, branch=branch)


def _download(url: str, dest: Path) -> None:
    print(f"  fetching {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "setup-dreamplace/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        with open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)


def _extract_tarball_into(tar_path: Path, target_dir: Path) -> None:
    """Extract a GitHub-style tarball (single top-level dir) into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        if not members:
            raise RuntimeError(f"empty tarball {tar_path}")
        top = members[0].name.split("/", 1)[0] + "/"
        for m in members:
            if not m.name.startswith(top):
                continue
            m.name = m.name[len(top):]
            if not m.name:
                continue
            tar.extract(m, target_dir)


def fetch_main(force: bool) -> None:
    if DPL_ROOT.exists() and not force:
        print(f"[1/5] DREAMPlace already present at {DPL_ROOT}, skipping fetch")
        return
    if force and DPL_ROOT.exists():
        shutil.rmtree(DPL_ROOT, ignore_errors=True)
    DPL_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tb = Path(td) / "dpl.tar.gz"
        _download(_tarball_url("limbo018/DREAMPlace", DPL_BRANCH), tb)
        _extract_tarball_into(tb, DPL_ROOT)
    print(f"[1/5] DREAMPlace extracted into {DPL_ROOT}")


def fetch_submodules(force: bool) -> None:
    tp = DPL_ROOT / "thirdparty"
    for name, user_repo, branch in SUBMODULES:
        target = tp / name
        if target.exists() and any(target.iterdir()) and not force:
            print(f"[2/5] submodule {name} present, skipping")
            continue
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        with tempfile.TemporaryDirectory() as td:
            tb = Path(td) / "sm.tar.gz"
            _download(_tarball_url(user_repo, branch), tb)
            _extract_tarball_into(tb, target)
        print(f"[2/5] submodule {name} fetched from {user_repo}@{branch}")

    # Limbo sub-submodules
    limbo_root = tp / "Limbo"
    for sub_rel, user_repo, branch, target_name in SUB_SUBMODULES:
        target = limbo_root / sub_rel
        if target.exists() and any(target.iterdir()) and not force:
            print(f"[2/5] sub-submodule {target_name} present, skipping")
            continue
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        with tempfile.TemporaryDirectory() as td:
            tb = Path(td) / "sm.tar.gz"
            _download(_tarball_url(user_repo, branch), tb)
            _extract_tarball_into(tb, target)
        print(f"[2/5] sub-submodule {target_name} fetched")


# --- Patches ----------------------------------------------------------------


def apply_patches() -> None:
    """Make DREAMPlace build with modern CMake + NumPy 2.

    Patches:
      A) limbo/thirdparty/lemon/CMakeLists.txt:
         - cmake_minimum_required 2.8 → 3.5
         - CMP0048 OLD → NEW
      B) limbo/parsers/CMakeLists.txt: comment out gdsii/gdsdb (needs quadmath.h)
      C) Create /tmp/qm_stub/quadmath.h (empty) so that boost cpp_bin_float can
         compile its transitive include without us pulling in libquadmath.
    """
    lemon_cml = (
        DPL_ROOT / "thirdparty/Limbo/limbo/thirdparty/lemon/CMakeLists.txt"
    )
    if lemon_cml.exists():
        txt = lemon_cml.read_text()
        new = txt
        new = new.replace("CMAKE_MINIMUM_REQUIRED(VERSION 2.8)",
                          "CMAKE_MINIMUM_REQUIRED(VERSION 3.5)")
        new = new.replace("CMAKE_POLICY(SET CMP0048 OLD)",
                          "CMAKE_POLICY(SET CMP0048 NEW)")
        if new != txt:
            lemon_cml.write_text(new)
            print(f"[3/5] patched lemon CMakeLists.txt (cmake_minimum_required + CMP0048)")
        else:
            print(f"[3/5] lemon CMakeLists.txt already patched")

    parsers_cml = DPL_ROOT / "thirdparty/Limbo/limbo/parsers/CMakeLists.txt"
    if parsers_cml.exists():
        txt = parsers_cml.read_text()
        if "add_subdirectory(gdsii/gdsdb)" in txt and "# add_subdirectory(gdsii/gdsdb)" not in txt:
            new = txt.replace("add_subdirectory(gdsii/gdsdb)",
                              "# add_subdirectory(gdsii/gdsdb) # disabled: needs quadmath.h")
            parsers_cml.write_text(new)
            print("[3/5] patched parsers CMakeLists.txt (disabled gdsdb)")
        else:
            print("[3/5] parsers CMakeLists.txt already patched")

    QM_STUB.mkdir(parents=True, exist_ok=True)
    qm_header = QM_STUB / "quadmath.h"
    if not qm_header.exists():
        qm_header.write_text("/* Stub quadmath.h to bypass boost cpp_bin_float transitive include. */\n"
                             "#ifndef QUADMATH_H\n#define QUADMATH_H\n#endif\n")
    print(f"[3/5] quadmath.h stub at {qm_header}")


# --- Build ------------------------------------------------------------------


def build() -> None:
    if (DPL_BUILD / "dreamplace" / "ops" / "place_io" /
            "place_io_cpp.cpython-310-x86_64-linux-gnu.so").exists():
        print("[4/5] place_io_cpp.so already built, skipping CMake build")
        return

    DPL_BUILD.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CPATH"] = f"{QM_STUB}:{env.get('CPATH', '')}"
    cmake_cmd = [
        "cmake", "..",
        "-DCMAKE_INSTALL_PREFIX=/tmp/dreamplace_install",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        "-DBOOST_ROOT=/opt/conda",
        "-DBoost_INCLUDE_DIR=/opt/conda/include",
        "-DZLIB_ROOT=/opt/conda",
        "-DZLIB_INCLUDE_DIR=/opt/conda/include",
        "-DZLIB_LIBRARY=/opt/conda/lib/libz.so",
        "-DBISON_EXECUTABLE=/opt/conda/bin/bison",
        "-DFLEX_EXECUTABLE=/opt/conda/bin/flex",
        "-DFLEX_INCLUDE_DIR=/opt/conda/include",
        f"-DPython_EXECUTABLE={VENV_PY}",
        f"-DPYTHON_EXECUTABLE={VENV_PY}",
        "-DCMAKE_CXX_ABI=1",
        "-Wno-dev",
    ]
    print("[4/5] running cmake...")
    subprocess.check_call(cmake_cmd, cwd=str(DPL_BUILD), env=env)

    nproc = str(os.cpu_count() or 8)
    # Build only the targets we need. Unittest targets pull in MKL libs that
    # aren't installed on this image; we skip them.
    targets = [
        "place_io_cpp", "electric_potential_cpp", "density_potential_cpp",
        "hpwl_cpp", "weighted_average_wirelength_cpp", "logsumexp_wirelength_cpp",
        "abacus_legalize_cpp", "greedy_legalize_cpp", "macro_legalize_cpp",
        "draw_place_cpp", "dct_cpp", "density_map_cpp", "density_overflow_cpp",
        "fence_region_cpp", "gift_init_cpp", "global_swap_cpp",
        "independent_set_matching_cpp", "independent_set_matching_sequential_cpp",
        "k_reorder_cpp", "legality_check_cpp", "move_boundary_cpp",
        "pin_pos_cpp", "pinrudy_cpp", "pin_utilization_cpp",
        "pin_weight_sum_cpp", "rudy_cpp", "timing_cpp", "adjust_node_area_cpp",
    ]
    print(f"[4/5] building {len(targets)} CPU targets with make -j{nproc} ...")
    subprocess.check_call(["make", "-j" + nproc, *targets], cwd=str(DPL_BUILD), env=env)


def install_py_sources() -> None:
    """Copy DREAMPlace's Python sources alongside the built .so files in build/."""
    src = DPL_ROOT / "dreamplace"
    dst = DPL_BUILD / "dreamplace"
    if not dst.exists():
        raise RuntimeError(f"build dir {dst} missing — build first")
    n = 0
    for p in src.rglob("*.py"):
        rel = p.relative_to(src)
        if rel.name == "configure.py":
            # configure.py is build-generated; don't clobber it.
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1
    # Copy params schema next to Params.py
    params_schema = src / "params.json"
    if params_schema.exists():
        shutil.copy2(params_schema, dst / "params.json")
    print(f"[5/5] installed {n} Python sources + params.json into {dst}")

    # NumPy 2.x compat patch: np.string_ → np.bytes_
    placedb = dst / "PlaceDB.py"
    if placedb.exists():
        t = placedb.read_text()
        if "np.string_" in t:
            placedb.write_text(t.replace("np.string_", "np.bytes_"))
            print("[5/5] patched PlaceDB.py for NumPy 2 (np.string_ → np.bytes_)")


# --- Verify -----------------------------------------------------------------


def verify() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(DPL_BUILD)
    check = (
        "import dreamplace.configure as c; "
        "import dreamplace.ops.place_io.place_io as pio; "
        "import dreamplace.ops.electric_potential.electric_potential as ep; "
        "print('OK:', list(c.compile_configurations)[:3], pio.__name__)"
    )
    res = subprocess.run([str(VENV_PY), "-c", check], env=env,
                         capture_output=True, text=True)
    if res.returncode != 0:
        print("VERIFY FAILED:")
        print(res.stdout)
        print(res.stderr)
        sys.exit(1)
    print(f"[verify] {res.stdout.strip()}")


# --- Main -------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-fetch and re-build even if artifacts exist")
    args = ap.parse_args()

    fetch_main(args.force)
    fetch_submodules(args.force)
    apply_patches()
    build()
    install_py_sources()
    verify()
    print("\nDREAMPlace ready. Import path:", DPL_BUILD)
    print("Use via submissions/hrt_winner/dpl_wrapper.py.")


if __name__ == "__main__":
    main()
