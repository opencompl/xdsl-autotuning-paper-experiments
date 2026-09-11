"""Run the whole evaluation on one Grid'5000 node, from one command.

`g5k-run` is general: it reserves a node and runs *a* command in a container on
it.  Everything specific to our evaluation -- which image, which node knobs,
where the source is staged, what the container is told about the node -- has to
be spelled out on its command line, which is how that invocation grew to
twenty lines that have to be retyped correctly for every machine.  This is
that command line, kept in one place:

    g5k-tools/.venv/bin/g5k-eval chirop-3
    g5k-tools/.venv/bin/g5k-eval --microarch "zen 5" --walltime 6:00:00
    g5k-tools/.venv/bin/g5k-eval chirop-3 --dry-run   # print the plan only

It also does the three things that invocation left to the caller, each of which
is a way to get a run wrong:

* it stages the working tree on the site's NFS home, with the revision in a
  file since `.git` does not travel, so the node runs *this* checkout rather
  than whatever was rsynced there last;
* it derives `PEAK` from the node the run will land on, instead of carrying
  the last machine's number over to a machine with different FMA pipes;
* it turns turbo off with whichever knob that node has -- `intel_pstate`'s
  `no_turbo` or `cpufreq`'s `boost` -- rather than the one the previous
  machine had.

Anything else goes to `g5k-run`: the options below are forwarded, and
`--run-arg=--poll=30` passes through one it does not name.

Credentials and SSH access are g5k-run's; see README.md.
"""

import argparse
import logging
import os
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

from enoslib.infra.enos_g5k.g5k_api_utils import get_api_username, get_cluster_site

from . import run as run_module
from .availability import client, describe_cpu, parse_walltime

logger = logging.getLogger("g5k-eval")

# The toolchain image is published from a `v*` tag (see
# .github/workflows/publish-docker.yml), so the tag the checkout describes to
# is the image built from this Dockerfile.
IMAGE_REPOSITORY = "ghcr.io/opencompl/xdsl-autotuning-ci"

# The evaluation is hours long on the machines worth running it on.
DEFAULT_WALLTIME = "6:00:00"

# Where the source waits on the site's home, and where the node stages it.
REMOTE_SOURCE = "xdsl-autotuning-paper-experiments"
NODE_SOURCE = "/tmp/eval"
CONTAINER_SOURCE = "/src"

# What identifies the repository root, and what runs inside the container.
MARKER = Path("scripts/g5k-eval.sh")
EVAL_COMMAND = ("bash", "scripts/g5k-eval.sh")
REVISION_FILE = ".g5k-revision"

# Everything the node does not need and that would make the upload
# minutes-long: the history, the two venvs, the build tree and the caches, and
# the results of previous runs.
STAGE_EXCLUDES = (
    ".git",
    ".venv",
    ".direnv",
    "build",
    ".snakemake",
    "g5k-runs",
    "__pycache__",
    ".pytest_cache",
)


# --------------------------------------------------------------------------- #
# This checkout
# --------------------------------------------------------------------------- #


def git(root: Path, *arguments: str) -> str | None:
    """Run a read-only git command in ``root``, or None if it fails."""
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def repository_root(explicit: str | None) -> Path:
    """The checkout to run: the one we are standing in, or ``--source``.

    Found by walking up from the working directory, so the command works from
    anywhere in the repository, and falling back to the checkout this tool was
    installed from -- g5k-tools lives inside the repository it runs.
    """
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / MARKER).is_file():
            raise SystemExit(f"error: {root} has no {MARKER}, so it is not the source")
        return root
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / MARKER).is_file():
            return candidate
    packaged = Path(__file__).resolve().parents[3]
    if (packaged / MARKER).is_file():
        return packaged
    raise SystemExit(
        f"error: no {MARKER} in this directory or above it; run this from the "
        "repository, or pass --source <checkout>"
    )


def revision(root: Path) -> str:
    """What to leave behind in `.g5k-revision`.

    The staging rsync excludes `.git`, and `scripts/g5k-eval.sh` copies this
    file into the results as the provenance of the numbers, so a dirty
    worktree is worth recording: the sha alone would claim more than the run
    can back up.
    """
    head = git(root, "rev-parse", "HEAD")
    if head is None:
        return "unknown (not a git checkout)"
    # The file we are about to write does not count as a change, or every run
    # after the first would report a dirty tree and the marker would mean
    # nothing.
    changed = [
        line
        for line in (git(root, "status", "--porcelain") or "").splitlines()
        if not line.endswith(REVISION_FILE)
    ]
    if changed:
        return f"{head} (worktree dirty when staged)"
    return head


def image_for(root: Path, explicit: str | None) -> str:
    """The toolchain image to run, from the tag this checkout describes to.

    The publish workflow tags the image with the semver of a `v*` git tag, so
    `v0.33.0` is `...:0.33.0`. That is the image built from *this* Dockerfile,
    which is the one whose toolchain matches the sources being staged.
    """
    if explicit:
        return explicit
    tag = git(root, "describe", "--tags", "--match", "v*", "--abbrev=0")
    if not tag:
        raise SystemExit(
            "error: no v* tag to name the image after (shallow clone, or no "
            "tags fetched); pass --image <repository:tag>"
        )
    return f"{IMAGE_REPOSITORY}:{tag.removeprefix('v')}"


# --------------------------------------------------------------------------- #
# This node
# --------------------------------------------------------------------------- #

# f32 FLOP/cycle -- vector lanes * FMA pipes * 2 -- which is what `PEAK` is and
# the one thing about a node that cannot be detected on it, so
# scripts/g5k-eval.sh refuses to start without it. Keyed on the reference API's
# microarchitecture, matched as a case-insensitive prefix exactly as
# `availability.matches` does, so "zen 4" also covers the dense "Zen 4c" parts.
# Same table as the one scripts/g5k-eval.sh prints when PEAK is missing.
TWO_FMA_PIPES = (
    "skylake-sp",
    "skylake-x",
    "cascade lake",
    "ice lake",
    "sapphire rapids",
    "emerald rapids",
    "granite rapids",
    "zen 5",
)
# AVX-512 over a 256-bit datapath: 512-bit instructions, half the throughput.
HALF_WIDTH_AVX512 = ("zen 4",)
# Intel sells the same microarchitecture with one 512-bit FMA pipe instead of
# two on its cheaper SKUs, and the reference API's `version` is where the SKU
# is ("Platinum 8358", "Gold 6130", "Silver 4310"). A wrong guess here is not
# an error anywhere downstream, just every % of peak figure at twice or half
# scale, so the chosen number is printed with the CPU it was chosen for.
ONE_FMA_PIPE = ("bronze", "silver", "gold 5")


def peak_for(microarchitecture: str | None, sku: str | None) -> tuple[int | None, str]:
    """This node's f32 FLOP/cycle, and the reasoning, or None if unknown."""
    name = (microarchitecture or "").strip().casefold()
    version = (sku or "").strip().casefold()
    if any(name.startswith(prefix) for prefix in HALF_WIDTH_AVX512):
        return 32, f"{microarchitecture}: AVX-512 over a 256-bit datapath"
    if any(name.startswith(prefix) for prefix in TWO_FMA_PIPES):
        if any(version.startswith(prefix) for prefix in ONE_FMA_PIPE):
            return 32, f"{microarchitecture} {sku}: one 512-bit FMA pipe on this SKU"
        return 64, f"{microarchitecture}: two 512-bit FMA pipes"
    return None, f"{microarchitecture or 'unknown microarchitecture'}: not in the table"


def node_hardware(
    site: str, cluster: str, uid: str | None
) -> tuple[str | None, str, str | None]:
    """``(microarchitecture, CPU, SKU)`` of a node, or of the cluster's first.

    A cluster's nodes are identical, so with ``--cluster`` -- where OAR has not
    picked a node yet -- any of them answers for all of them.
    """
    for node in client().sites[site].clusters[cluster].nodes.list():
        if uid is not None and node.uid != uid:
            continue
        processor = getattr(node, "processor", None) or {}
        architecture = getattr(node, "architecture", None) or {}
        return (
            processor.get("microarchitecture"),
            describe_cpu(processor, architecture),
            processor.get("version"),
        )
    return None, "unknown CPU", None


# --------------------------------------------------------------------------- #
# Getting the source to the node
# --------------------------------------------------------------------------- #


def remote_path(site: str, remote_source: str) -> str:
    """Where the source lives, addressed from access.grid5000.fr.

    Home directories are per site, and access exports each site's home under
    its own name, so a site-relative path is the one string that works both
    from access and (as ``$HOME/...``) from the node.
    """
    if remote_source.startswith("/"):
        return remote_source
    return f"{site}/{remote_source}"


def stage_source(
    root: Path, login: str, site: str, remote_source: str, dry_run: bool
) -> None:
    """Copy this checkout to the site's home, where the node can read it.

    The image is only the toolchain -- the source is bind-mounted into it, as
    `make docker-run` does -- so the source has to be somewhere the node can
    see, and that is the site's NFS home. Excluded paths are left alone on the
    far side rather than deleted, which is what keeps a `build/` tree there
    across runs.
    """
    path = remote_path(site, remote_source)
    (root / REVISION_FILE).write_text(revision(root) + "\n")
    command = [
        "rsync",
        "-az",
        "--delete",
        "--info=progress2",
        *(f"--exclude={pattern}" for pattern in STAGE_EXCLUDES),
        f"{root}/",
        f"{login}@{run_module.ACCESS}:{path}/",
    ]
    if dry_run:
        print(f"would stage:  {shlex.join(command)}")
        return
    logger.info("staging %s on %s:%s", root, site, path)
    run_module.ssh_access(login, f"mkdir -p {shlex.quote(path)}")
    subprocess.run(command, check=True)


# The clock, and the two ways a node spells "no turbo". Intel nodes driven by
# intel_pstate have no_turbo; everything else (acpi-cpufreq, amd-pstate) has
# boost, with the opposite sense. Deciding on the node rather than here keeps
# this one string right for every machine.
TURBO_OFF = (
    "if [ -e /sys/devices/system/cpu/intel_pstate/no_turbo ]; then "
    "echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null; "
    "elif [ -e /sys/devices/system/cpu/cpufreq/boost ]; then "
    "echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost >/dev/null; "
    "else echo '[g5k-eval] no turbo knob on this node'; fi"
)


def node_setup(remote_source: str) -> list[str]:
    """What has to happen on the node before the container starts."""
    source = (
        remote_source if remote_source.startswith("/") else f"$HOME/{remote_source}"
    )
    return [
        # Not namespaced, so no container can write it however privileged it
        # is. Without it PAPI cannot read PAPI_TOT_CYC and the harness times
        # with the monotonic clock scaled by a nominal frequency -- while the
        # datasets record cycles and `peak` is per cycle.
        "sudo sysctl -w kernel.perf_event_paranoid=-1",
        "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null",
        TURBO_OFF,
        # The container runs as root, so everything it left here last time is
        # root's, and this rsync -- which runs as us -- cannot delete it.
        f"sudo chown -R $(id -u):$(id -g) {NODE_SOURCE} || true",
        # Node-local disk: `build/` is tens of thousands of small files, and
        # writing that from a hundred parallel jobs onto the site's NFS server
        # is both slow and antisocial. `--exclude build` keeps the build cache
        # across runs, which is safe because every artifact is keyed on a
        # digest of the generator that produced it.
        f"rsync -a --delete --exclude build {source}/ {NODE_SOURCE}/",
    ]


# --------------------------------------------------------------------------- #
# The g5k-run invocation
# --------------------------------------------------------------------------- #


def run_arguments(
    *,
    target: run_module.Target,
    image: str,
    peak: int,
    walltime: str,
    remote_source: str,
    keep: bool,
    forwarded: Sequence[str] = (),
) -> list[str]:
    """The g5k-run command line for an evaluation run on ``target``."""
    arguments = ["--walltime", walltime]
    if target.server:
        arguments += ["--server", target.server]
    else:
        arguments += ["--cluster", str(target.cluster)]
    if keep:
        # The run is hours long: holding the node means a failure late in it
        # costs neither the reservation nor the warm image, and the retry lands
        # on the same node, so the second attempt is comparable to the first.
        arguments.append("--keep")
    arguments += ["--pull", "--image", image]
    for command in node_setup(remote_source):
        arguments += ["--node-setup", command]
    arguments += ["--mount", f"{NODE_SOURCE}:{CONTAINER_SOURCE}"]
    arguments += ["--workdir", CONTAINER_SOURCE]
    arguments += [
        "--env",
        "IN_DOCKER=1",
        "--env",
        "SNAKEMAKE_SCHEDULER=greedy",
        "--env",
        f"PEAK={peak}",
    ]
    # PAPI needs the counters and the host's view of them; seccomp stands in
    # the way of the perf syscalls. Attached, so argparse does not read them as
    # options of g5k-run's own.
    arguments += [
        "--docker-arg=--cap-add=SYS_ADMIN",
        "--docker-arg=--cap-add=PERFMON",
        "--docker-arg=--security-opt",
        "--docker-arg=seccomp=unconfined",
        "--docker-arg=--pid=host",
    ]
    # Forwarded options come last, so a caller's `--env PEAK=32` or
    # `--env VALIDATE=0` overrides ours: docker keeps the last `-e` for a name.
    arguments += list(forwarded)
    return arguments + ["--", *EVAL_COMMAND]


def forwarded_arguments(args: argparse.Namespace) -> list[str]:
    """The g5k-run options this wrapper passes straight through."""
    forwarded: list[str] = []
    for name in ("queue", "reservation", "job_name", "fetch_to"):
        value = getattr(args, name)
        if value:
            forwarded += [f"--{name.replace('_', '-')}", value]
    for flag in ("wait", "dry_run", "no_fetch", "verbose"):
        if getattr(args, flag):
            forwarded.append(f"--{flag.replace('_', '-')}")
    for variable in args.env:
        forwarded += ["--env", variable]
    forwarded += args.run_arg
    return forwarded


def selection(args: argparse.Namespace) -> argparse.Namespace:
    """The node selection, in the shape `run.resolve_target` reads.

    The target is resolved here rather than left to g5k-run because the source
    has to be staged on the *site's* home before the job starts, and with
    `--microarch` the site is only known once a node has been picked.
    """
    return argparse.Namespace(
        server=args.node,
        cluster=args.cluster,
        microarch=args.microarch,
        sites=args.sites,
        exclude_sites=args.exclude_sites,
        exclude_nodes=args.exclude_nodes,
        queue=args.queue,
        wait=args.wait,
        reservation=args.reservation,
        workers=8,
    )


def names(target: run_module.Target) -> tuple[str, str | None]:
    """``(cluster, node uid)`` of a target; the uid is None for a cluster."""
    if target.server:
        uid = target.server.split(".")[0]
        return uid.rsplit("-", 1)[0], uid
    return str(target.cluster), None


def latest_run(label: str, fetch_to: str | None) -> Path | None:
    """The run directory g5k-run just fetched, if it is where we expect it."""
    runs = Path(fetch_to or "g5k-runs")
    if not runs.is_dir():
        return None
    fetched = sorted(path for path in runs.glob(f"*-{label}") if path.is_dir())
    return fetched[-1] if fetched else None


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="g5k-eval",
        description=(
            "Run the evaluation on one Grid'5000 node: stage this checkout on "
            "the site's home, reserve the node, and run scripts/g5k-eval.sh "
            "in the toolchain image on it."
        ),
        epilog=(
            "Unnamed g5k-run options go through --run-arg=--poll=30. "
            "Eval knobs are the container's environment: --env VALIDATE=0, "
            "--env CORES=32, --env PIN_CPU=2."
        ),
    )

    selection_group = parser.add_argument_group("node selection")
    exclusive = selection_group.add_mutually_exclusive_group()
    exclusive.add_argument(
        "node",
        nargs="?",
        help=(
            "the node to run on: chirop-3, chirop-3.lille or the full name. "
            "Without a site we ask the reference API which site the cluster "
            "is at."
        ),
    )
    exclusive.add_argument(
        "--cluster", metavar="NAME", help="any node of this cluster instead"
    )
    exclusive.add_argument(
        "--microarch",
        metavar="NAME",
        help=(
            "the soonest free node with this microarchitecture instead; see "
            "g5k-availability"
        ),
    )
    selection_group.add_argument(
        "--sites", nargs="+", metavar="SITE", help="only search these sites"
    )
    selection_group.add_argument(
        "--exclude-sites",
        nargs="+",
        default=[],
        metavar="SITE",
        help="skip these sites",
    )
    selection_group.add_argument(
        "--exclude-nodes",
        nargs="+",
        default=[],
        metavar="NODE",
        help="skip these nodes",
    )
    selection_group.add_argument("--queue", help="OAR queue (default: the node's own)")
    selection_group.add_argument(
        "--wait", action="store_true", help="queue the job even if nothing is free now"
    )

    job = parser.add_argument_group("reservation")
    job.add_argument(
        "--walltime",
        default=DEFAULT_WALLTIME,
        help="job walltime (default: %(default)s)",
    )
    job.add_argument(
        "--reservation", metavar="'YYYY-MM-DD HH:MM:SS'", help="book this date instead"
    )
    job.add_argument("--job-name", help="OAR job name (default: g5k-run-<node>)")
    job.add_argument(
        "--no-keep",
        action="store_true",
        help="give the node back when the run ends instead of holding it",
    )

    evaluation = parser.add_argument_group("the run")
    evaluation.add_argument(
        "--image",
        help=f"toolchain image (default: {IMAGE_REPOSITORY}:<this checkout's v* tag>)",
    )
    evaluation.add_argument(
        "--peak",
        type=int,
        metavar="N",
        help=(
            "this node's f32 FLOP/cycle, overriding what its microarchitecture "
            "implies; 0 records none and leaves the %% of peak figures "
            "undrawable"
        ),
    )
    evaluation.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="K=V",
        help="environment variable for the container, repeatable",
    )
    evaluation.add_argument(
        "--source", metavar="DIR", help="checkout to run (default: the one we are in)"
    )
    evaluation.add_argument(
        "--remote-source",
        default=REMOTE_SOURCE,
        metavar="PATH",
        help="where to stage it on the site's home (default: %(default)s)",
    )
    evaluation.add_argument(
        "--no-stage",
        action="store_true",
        help="run what is already on the site's home, staging nothing",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--fetch-to",
        metavar="DIR",
        help="local directory for the run (default: g5k-runs)",
    )
    output.add_argument(
        "--no-fetch", action="store_true", help="leave the results on Grid'5000"
    )
    output.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the node script, reserve and stage nothing",
    )
    output.add_argument("--verbose", action="store_true", help="debug logging")
    output.add_argument(
        "--run-arg",
        action="append",
        default=[],
        metavar="ARG",
        help=(
            "extra g5k-run argument, repeatable. Write it attached, as "
            "--run-arg=--poll=30."
        ),
    )
    return parser.parse_args(argv)


def resolve_node(name: str) -> str:
    """Accept a bare ``chirop-3`` next to g5k-run's ``<node>.<site>`` forms."""
    if "." in name:
        return run_module.normalize_server(name)
    cluster = name.rsplit("-", 1)[0]
    try:
        site = get_cluster_site(cluster)
    except Exception as error:
        raise SystemExit(
            f"error: cannot find which site {cluster!r} is at ({error}); "
            f"pass the node as {name}.<site>"
        ) from None
    return f"{name}.{site}.grid5000.fr"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s"
    )

    if not (args.node or args.cluster or args.microarch):
        return run_module.fail(
            "say which node to run on: a node name (chirop-3), --cluster, or "
            "--microarch. There is no default, because the machine decides "
            "what the run measures."
        )

    try:
        root = repository_root(args.source)
        walltime = parse_walltime(args.walltime)
        if args.node:
            args.node = resolve_node(args.node)
    except (ValueError, SystemExit) as error:
        return run_module.fail(str(error).removeprefix("error: "))

    try:
        login = get_api_username()
    except Exception as error:
        return run_module.fail(f"cannot read your Grid'5000 login ({error})")
    if not login:
        return run_module.fail(
            "no Grid'5000 login found; set it in ~/.python-grid5000.yaml or in G5K_USER"
        )

    # A cluster or node that does not exist reaches the reference API as a
    # lookup that raises rather than answers, and an enoslib KeyError is not a
    # useful thing to print at someone.
    try:
        target = run_module.resolve_target(selection(args), walltime)
        cluster, uid = names(target)
        microarchitecture, cpu, sku = node_hardware(target.site, cluster, uid)
    except SystemExit as error:
        return run_module.fail(str(error).removeprefix("error: "))
    except Exception as error:
        return run_module.fail(
            f"cannot resolve {args.node or args.cluster or args.microarch!r} "
            f"({type(error).__name__}: {error}); g5k-availability lists the "
            "nodes that exist"
        )

    if args.peak is not None:
        peak, why = args.peak, "given with --peak"
    else:
        peak, why = peak_for(microarchitecture, sku)
    if peak is None:
        return run_module.fail(
            f"cannot tell this node's f32 FLOP/cycle ({why}): pass --peak "
            "<lanes * FMA pipes * 2>, which is 64 on the Intel AVX-512 parts "
            "with two FMA pipes and on Zen 5, and 32 on Zen 4, Zen 4c and the "
            "one-pipe Intel SKUs. --peak 0 runs anyway and records none, "
            "leaving the % of peak figures undrawable."
        )

    try:
        image = image_for(root, args.image)
    except SystemExit as error:
        return run_module.fail(str(error).removeprefix("error: "))

    # The run directory is fetched relative to the working directory, and the
    # results belong next to the data they will be folded into.
    if Path.cwd().resolve() != root:
        logger.info("working from %s", root)
        os.chdir(root)

    print(f"source:   {root} @ {revision(root)}")
    print(f"image:    {image}")
    print(f"machine:  {cluster} ({cpu})")
    print(f"peak:     PEAK={peak}  [{why}]")
    print()

    # g5k-run checks this too, but only once it is about to reserve -- which
    # is after the upload. Staging a busy node's source is minutes of rsync
    # thrown away, so ask before rather than after.
    job_name = args.job_name or f"g5k-run-{target.label}"
    try:
        run_module.verify_free(
            target,
            walltime,
            args.wait or bool(args.reservation),
            job_name,
            login,
        )
    except SystemExit as error:
        return run_module.fail(str(error))

    if args.no_stage:
        logger.info(
            "not staging; the run uses whatever is in %s:%s",
            target.site,
            remote_path(target.site, args.remote_source),
        )
    else:
        try:
            stage_source(root, login, target.site, args.remote_source, args.dry_run)
        except subprocess.CalledProcessError as error:
            return run_module.fail(
                f"staging the source failed (rsync {error.returncode})"
            )

    exit_code = run_module.main(
        run_arguments(
            target=target,
            image=image,
            peak=peak,
            walltime=args.walltime,
            remote_source=args.remote_source,
            keep=not args.no_keep,
            forwarded=forwarded_arguments(args),
        )
    )

    if exit_code == 0 and not (args.dry_run or args.no_fetch):
        run = latest_run(target.label, args.fetch_to) or "g5k-runs/<run_id>"
        print()
        print("to fold this run into the repository:")
        print(f"  cp {run}/results/{cluster}.json machines/")
        print(
            f"  mkdir -p data/{cluster} && "
            f"cp {run}/results/data/*.jsonl data/{cluster}/"
        )
        print(f"  make plots-machine MACHINE={cluster}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
