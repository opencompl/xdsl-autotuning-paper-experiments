"""Tests for the parts of g5k-eval that do not talk to Grid'5000.

The interesting decisions it makes are all pure: which image this checkout
implies, what a node's `PEAK` is, and the g5k-run command line it composes.
"""

import pytest

from g5k_tools.eval import (
    CONTAINER_SOURCE,
    EVAL_COMMAND,
    IMAGE_REPOSITORY,
    NODE_SOURCE,
    forwarded_arguments,
    image_for,
    node_setup,
    parse_args,
    peak_for,
    remote_path,
    repository_root,
    revision,
    run_arguments,
)
from g5k_tools.run import Target, parse_args as run_parse_args

NODE = Target(site="lille", queue="default", server="chirop-3.lille.grid5000.fr")


def arguments(**kwargs) -> list[str]:
    defaults = dict(
        target=NODE,
        image="img:1",
        peak=64,
        walltime="6:00:00",
        remote_source="xdsl-autotuning-paper-experiments",
        keep=True,
    )
    defaults.update(kwargs)
    return run_arguments(**defaults)


# --------------------------------------------------------------------------- #
# PEAK
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "microarchitecture,sku,expected",
    [
        # The machines already run: grvingt and chirop.
        ("Skylake-SP", "Gold 6130", 64),
        ("Ice Lake-SP", "Platinum 8358", 64),
        ("Emerald Rapids", "Platinum 8568Y+", 64),
        # Zen 4's AVX-512 is 256 bits wide underneath, and "zen 4" also has to
        # cover the dense Zen 4c parts.
        ("Zen 4", "EPYC 9254", 32),
        ("Zen 4c", "EPYC 9754", 32),
        ("Zen 5", "EPYC 9535", 64),
        # One 512-bit FMA pipe on the cheaper Intel SKUs.
        ("Cascade Lake", "Silver 4214", 32),
        ("Skylake-SP", "Bronze 3104", 32),
        ("Cascade Lake", "Gold 5220", 32),
    ],
)
def test_peak_follows_the_microarchitecture_and_the_sku(
    microarchitecture, sku, expected
):
    peak, why = peak_for(microarchitecture, sku)
    assert peak == expected
    assert microarchitecture.casefold() in why.casefold()


@pytest.mark.parametrize("microarchitecture", [None, "Broadwell", "Zen 2"])
def test_peak_is_unknown_rather_than_guessed(microarchitecture):
    # A pre-AVX-512 machine cannot generate our variants at all, so there is no
    # number to fall back on: g5k-eval asks for --peak instead.
    assert peak_for(microarchitecture, "whatever")[0] is None


# --------------------------------------------------------------------------- #
# This checkout
# --------------------------------------------------------------------------- #


def checkout(tmp_path, monkeypatch):
    """A directory that looks like the repository, seen from a subdirectory."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "g5k-eval.sh").write_text("#!/bin/sh\n")
    deep = tmp_path / "plots" / "nested"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    return tmp_path


def test_repository_root_is_found_from_anywhere_inside_it(tmp_path, monkeypatch):
    root = checkout(tmp_path, monkeypatch)
    assert repository_root(None) == root.resolve()
    assert repository_root(str(root)) == root.resolve()


def test_repository_root_rejects_a_directory_that_is_not_the_source(tmp_path):
    with pytest.raises(SystemExit):
        repository_root(str(tmp_path))


def test_revision_and_image_need_a_checkout(tmp_path, monkeypatch):
    root = checkout(tmp_path, monkeypatch)
    # Not a git checkout: the revision says so rather than claiming a sha, and
    # the image cannot be named after a tag that does not exist.
    assert "unknown" in revision(root)
    with pytest.raises(SystemExit):
        image_for(root, None)
    assert image_for(root, "img:2") == "img:2"


def test_image_is_named_after_the_tag_the_checkout_describes_to(tmp_path, monkeypatch):
    root = checkout(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "g5k_tools.eval.git",
        lambda _root, *arguments: "v0.33.0" if arguments[0] == "describe" else "",
    )
    assert image_for(root, None) == f"{IMAGE_REPOSITORY}:0.33.0"


# --------------------------------------------------------------------------- #
# The g5k-run command line
# --------------------------------------------------------------------------- #


def test_the_composed_command_line_is_one_g5k_run_accepts():
    parsed = run_parse_args(arguments())
    assert parsed.server == "chirop-3.lille.grid5000.fr"
    assert parsed.image == "img:1"
    assert parsed.pull and parsed.keep
    assert parsed.walltime == "6:00:00"
    assert parsed.workdir == CONTAINER_SOURCE
    assert f"{NODE_SOURCE}:{CONTAINER_SOURCE}" in parsed.mount
    assert "PEAK=64" in parsed.env
    assert parsed.command == ["--", *EVAL_COMMAND]
    # The `docker run` arguments have to survive argparse attached.
    assert "--cap-add=PERFMON" in parsed.docker_arg
    assert "--pid=host" in parsed.docker_arg


def test_a_cluster_target_reserves_any_node_of_it():
    parsed = run_parse_args(
        arguments(target=Target(site="nancy", queue="default", cluster="grvingt"))
    )
    assert (parsed.cluster, parsed.server) == ("grvingt", None)


def test_no_keep_gives_the_node_back():
    assert not run_parse_args(arguments(keep=False)).keep


def test_the_node_is_prepared_for_the_things_a_container_cannot_do():
    setup = node_setup("xdsl-autotuning-paper-experiments")
    joined = "\n".join(setup)
    # Not namespaced, so it cannot happen in the container.
    assert "kernel.perf_event_paranoid=-1" in joined
    # Both spellings of "no turbo", decided on the node.
    assert "intel_pstate/no_turbo" in joined and "cpufreq/boost" in joined
    # The staging rsync runs as us over what the container left as root.
    assert (
        setup.index(f"sudo chown -R $(id -u):$(id -g) {NODE_SOURCE} || true")
        < len(setup) - 1
    )
    assert setup[-1] == (
        "rsync -a --delete --exclude build "
        f"$HOME/xdsl-autotuning-paper-experiments/ {NODE_SOURCE}/"
    )


def test_an_absolute_remote_source_is_not_put_under_home():
    assert node_setup("/data/eval")[-1].endswith(f"/data/eval/ {NODE_SOURCE}/")
    assert remote_path("lille", "/data/eval") == "/data/eval"
    assert remote_path("lille", "eval") == "lille/eval"


def test_forwarded_options_come_after_ours_so_they_win():
    args = parse_args(
        ["chirop-3", "--env", "PEAK=32", "--env", "VALIDATE=0", "--run-arg=--poll=30"]
    )
    forwarded = forwarded_arguments(args)
    assert forwarded == [
        "--env",
        "PEAK=32",
        "--env",
        "VALIDATE=0",
        "--poll=30",
    ]
    parsed = run_parse_args(arguments(forwarded=forwarded[:4]))
    # docker keeps the last -e for a name, so a caller can override PEAK.
    assert parsed.env.index("PEAK=32") > parsed.env.index("PEAK=64")


def test_the_selection_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["chirop-3", "--microarch", "zen 5"])
