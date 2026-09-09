"""Tests for the pieces of g5k-run that do not talk to Grid'5000.

Everything here is about the strings we send to a node: the node script, the
`docker run` line and the log-polling protocol. They need no credentials.
"""

import base64

import pytest

from g5k_tools.availability import preferred_queue
from g5k_tools.run import (
    container_command,
    docker_arguments,
    make_layout,
    normalize_server,
    parse_args,
    parse_polled,
    render_script,
    resolve_target,
)

LAYOUT = make_layout("alice", "g5k-run", "20260101-000000-gres-1")


def script(**kwargs) -> str:
    defaults = dict(
        layout=LAYOUT,
        image="",
        image_tar=None,
        pull=False,
        docker_args=[],
        command=["true"],
    )
    defaults.update(kwargs)
    return render_script(**defaults)


# --------------------------------------------------------------------------- #
# Paths and node names
# --------------------------------------------------------------------------- #


def test_layout_lives_under_the_site_home():
    assert LAYOUT.rundir == "/home/alice/g5k-run/20260101-000000-gres-1"
    assert LAYOUT.log.endswith("/run.log")
    assert LAYOUT.results.endswith("/results")
    # Home-relative, so the same path also addresses access.grid5000.fr.
    assert str(LAYOUT.relative) == "g5k-run/20260101-000000-gres-1"


@pytest.mark.parametrize(
    "given,expected",
    [
        ("gres-1.nancy", "gres-1.nancy.grid5000.fr"),
        ("gres-1.nancy.grid5000.fr", "gres-1.nancy.grid5000.fr"),
    ],
)
def test_normalize_server(given, expected):
    assert normalize_server(given) == expected


def test_normalize_server_needs_a_site():
    with pytest.raises(ValueError):
        normalize_server("gres-1")


def test_resolve_target_reads_site_and_cluster_from_the_node():
    args = parse_args(
        ["--server", "gres-1.nancy", "--queue", "production", "--", "true"]
    )
    target = resolve_target(args, 3600)
    assert (target.site, target.queue, target.server) == (
        "nancy",
        "production",
        "gres-1.nancy.grid5000.fr",
    )
    assert target.label == "gres-1"


def test_preferred_queue_prefers_default_then_production():
    assert preferred_queue(["besteffort", "production"]) == "production"
    assert preferred_queue(["besteffort", "default", "production"]) == "default"
    assert preferred_queue([]) == "unknown"


# --------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------- #


def test_container_command_drops_the_separator():
    assert container_command(["--", "make", "dataset"]) == ["make", "dataset"]
    assert container_command([]) == []


def test_docker_arguments_always_mount_the_results_directory():
    args = parse_args(
        [
            "--mount",
            "/home/alice/repo:/src",
            "--env",
            "IN_DOCKER=1",
            "--workdir",
            "/src",
            "--docker-arg=--cap-add=PERFMON",
            "--",
            "true",
        ]
    )
    assert docker_arguments(args, LAYOUT) == [
        "-v",
        f"{LAYOUT.results}:/results",
        "-v",
        "/home/alice/repo:/src",
        "-e",
        "IN_DOCKER=1",
        "-w",
        "/src",
        "--cap-add=PERFMON",
    ]


def test_the_command_is_quoted_for_the_remote_shell():
    body = script(image="alpine", pull=True, command=["bash", "-c", "echo $HOME > x"])
    assert "docker run --rm --name 20260101-000000-gres-1" in body
    # The whole shell snippet must survive as one argument.
    assert """'echo $HOME > x'""" in body


# --------------------------------------------------------------------------- #
# The node script
# --------------------------------------------------------------------------- #


def test_the_script_always_leaves_an_exit_code_behind():
    # The poller waits for this file, so a setup failure has to write it too.
    assert f"> {LAYOUT.exit_code}" in script()
    assert "trap" in script()


def test_docker_is_installed_only_when_missing():
    body = script(image="alpine", pull=True)
    assert "command -v docker" in body
    assert "g5k-setup-docker -t" in body


def test_a_tarball_is_loaded_and_can_name_its_own_image():
    body = script(image_tar="/home/alice/g5k-run/images/img.tar")
    assert "docker load -i /home/alice/g5k-run/images/img.tar" in body
    assert "sed -n 's/^Loaded image: //p'" in body
    assert "docker pull" not in body


def test_pull_mode_pulls_the_named_image():
    body = script(image="alpine:3", pull=True)
    assert "image=alpine:3" in body
    assert "docker pull" in body
    assert "docker load" not in body


def test_without_a_source_the_image_must_already_be_there():
    body = script(image="alpine:3")
    assert "is not on this node" in body
    assert "docker pull" not in body
    assert "docker load" not in body


def test_node_setup_runs_before_the_image_is_fetched():
    # Ordering matters twice: a setup that fails should not first cost a
    # multi-gigabyte pull, and the sysctl has to be in place before the
    # container that needs it starts.
    body = script(
        image="alpine:3",
        pull=True,
        node_setup=["sudo sysctl -w kernel.perf_event_paranoid=-1"],
    )
    assert "sudo sysctl -w kernel.perf_event_paranoid=-1" in body
    assert body.index("perf_event_paranoid") < body.index("docker pull")
    assert body.index("g5k-setup-docker") < body.index("perf_event_paranoid")


def test_node_setup_commands_keep_their_shell_meaning_and_are_echoed():
    setup = "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/x"
    body = script(image="alpine:3", pull=True, node_setup=[setup])
    # Verbatim: globs, pipes and $HOME are the point of these commands.
    assert f"\n{setup}\n" in body
    # And announced, so run.log says which step failed.
    assert "[g5k-run] setup: echo performance" in body


def test_node_setup_runs_in_the_order_given():
    body = script(image="alpine:3", pull=True, node_setup=["first", "second"])
    assert body.index("\nfirst\n") < body.index("\nsecond\n")


def test_no_node_setup_leaves_the_script_alone():
    assert "--node-setup" not in script(image="alpine:3", pull=True)


def test_the_container_is_told_which_node_it_is_on():
    # A container's hostname is its own id, so a command that configures itself
    # from the hardware cannot otherwise tell where it landed. Expanded on the
    # node, since with --cluster the node is only settled once OAR has picked.
    body = script(image="alpine:3", pull=True)
    assert "cluster=${short%-*}" in body
    for variable in ("G5K_NODE", "G5K_CLUSTER", "G5K_SITE", "G5K_JOB_ID"):
        assert f'-e {variable}="' in body


# --------------------------------------------------------------------------- #
# Polling
# --------------------------------------------------------------------------- #


def polled(chunk: bytes, code: str = "") -> str:
    encoded = base64.b64encode(chunk).decode()
    return f"@@LOG@@{encoded}@@RC@@{code}@@END@@"


def test_parse_polled_returns_exact_bytes_while_running():
    # Exact bytes matter: the next read resumes at this many bytes in.
    assert parse_polled(polled(b"h\xc3\xa9llo\n")) == (b"h\xc3\xa9llo\n", None)
    assert parse_polled(polled(b"")) == (b"", None)


def test_parse_polled_reports_the_exit_code_once_it_is_there():
    assert parse_polled(polled(b"done\n", "0\n")) == (b"done\n", "0")
    assert parse_polled(polled(b"", "137\n")) == (b"", "137")


def test_parse_polled_waits_for_a_whole_exit_code():
    # The number and its newline are two writes; a poll can land between them.
    assert parse_polled(polled(b"", "13")) == (b"", None)


def test_parse_polled_ignores_whatever_the_shell_says_around_it():
    banner = "Linux gres-1 6.1.0\n" + polled(b"out") + "\nConnection closed\n"
    assert parse_polled(banner) == (b"out", None)


def test_parse_polled_survives_a_truncated_read():
    assert parse_polled("@@LOG@@partial") == (b"", None)
