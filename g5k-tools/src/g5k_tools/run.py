"""Run a command in a Docker container on a reserved Grid'5000 node.

One invocation does the whole round trip: pick a node, reserve it through
enoslib (OAR), install Docker on it with `g5k-setup-docker -t`, make the image
available, run the command in a container, stream its output here, and rsync
the run directory back.

    cd g5k-tools
    # look before you leap: prints the plan and the node script, reserves nothing
    uv run g5k-run --dry-run --microarch "zen 4" -- lscpu

    # pull the toolchain image from ghcr and run something in it
    uv run g5k-run --microarch "zen 4" --pull \
        --image ghcr.io/opencompl/xdsl-autotuning-ci:0.31.0 \
        -- bash -lc 'lscpu | head'

    # or ship a local image through the site's NFS home (no registry involved)
    uv run g5k-run --server gres-1.nancy.grid5000.fr \
        --image-tar ~/xdsl-autotuner.tar --image xdsl-autotuner \
        --mount /home/$USER/xdsl-autotuning-paper-experiments:/src \
        -- make dataset

    # the evaluation, on whichever node of that microarchitecture is free
    uv run g5k-run --microarch "zen 5" --walltime 6:00:00 --keep \
        --pull --image ghcr.io/opencompl/xdsl-autotuning-ci:0.32.0 \
        --node-setup 'sudo sysctl -w kernel.perf_event_paranoid=-1' \
        --mount /tmp/eval:/src --workdir /src \
        -- bash scripts/g5k-eval.sh

``--node-setup`` runs commands on the node before the container, for the things
a container cannot do for itself: writing a sysctl that is not namespaced
(``kernel.perf_event_paranoid``, without which PAPI cannot read cycle
counters), fixing the CPU clock, or staging data on node-local disk. The
container is told where it is running through ``G5K_NODE``, ``G5K_CLUSTER``,
``G5K_SITE`` and ``G5K_JOB_ID``, since a container's own hostname is its id.

Reservations are named (``--job-name``, by default derived from the node) and
enoslib *reloads* a job of that name instead of submitting a second one, so a
re-run lands on the same node rather than piling up reservations. The job is
deleted as soon as the command finishes, unless ``--keep`` is given to hold the
node for the next run (Docker and the loaded image then stay warm).

Everything on the node runs as your Grid'5000 user, not root, so `$HOME` is the
site's NFS home and the files the container writes stay yours.

Credentials are g5k-availability's: ~/.python-grid5000.yaml or G5K_USER /
G5K_PASSWORD. Talking to the nodes also needs the usual SSH access to
access.grid5000.fr; enoslib jumps through it automatically from outside.
"""

import argparse
import base64
import json
import logging
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from collections.abc import Sequence

import enoslib as en
from enoslib.log import DisableLogging
from enoslib.infra.enos_g5k.g5k_api_utils import (
    get_api_username,
    get_cluster_site,
    grid_reload_jobs_from_name,
)

from .availability import (
    RESERVABLE_STATES,
    Machine,
    annotate_availability,
    client,
    discover,
    format_when,
    parse_walltime,
    preferred_queue,
    soonest_start,
    sort_key,
)

logger = logging.getLogger("g5k-run")

# The one machine every Grid'5000 user can reach from the outside; both the
# SSH jump enoslib sets up and our own file transfers go through it.
ACCESS = "access.grid5000.fr"

# Wrapping the polled output in markers keeps us immune to anything the login
# shell might print around it, and base64 keeps the bytes exact so the offset
# we resume from stays right.
POLLED = re.compile(r"@@LOG@@(.*)@@RC@@(.*)@@END@@", re.DOTALL)

ROLE = "node"


# --------------------------------------------------------------------------- #
# Choosing a node
# --------------------------------------------------------------------------- #


@dataclass
class Target:
    """The node(s) to reserve, as an OAR selection."""

    site: str
    queue: str
    # Exactly one of the two: an explicit node, or one node from a cluster.
    server: str | None = None
    cluster: str | None = None
    description: str = ""
    # True when the node was chosen *because* it is free; an explicitly named
    # node or cluster still has to be checked.
    verified: bool = False

    @property
    def label(self) -> str:
        """Short name of the target, used to name the job and the run."""
        if self.server:
            return self.server.split(".")[0]
        return str(self.cluster)


def normalize_server(name: str) -> str:
    """Accept ``gres-1``-style shorthands next to full node names.

    enoslib reads the site out of the node name, so it needs at least
    ``<uid>.<site>``; anything shorter cannot be resolved here.
    """
    parts = name.split(".")
    if len(parts) == 2:
        return f"{name}.grid5000.fr"
    if len(parts) == 4:
        return name
    raise ValueError(
        f"invalid node name {name!r}: use <node>.<site> or <node>.<site>.grid5000.fr"
    )


def cluster_queue(site: str, cluster: str) -> str:
    """The OAR queue a cluster's nodes live in.

    Production clusters are not in the default queue, and a reservation that
    omits ``-q production`` for them is simply refused.
    """
    for node in client().sites[site].clusters[cluster].nodes.list():
        queues = (getattr(node, "supported_job_types", None) or {}).get("queues") or []
        if queues:
            return preferred_queue(queues)
    return "default"


def pick_machine(
    microarch: str,
    sites: Sequence[str] | None,
    exclude_sites: Sequence[str],
    exclude_nodes: Sequence[str],
    walltime: int,
    workers: int,
    wait: bool,
) -> Machine:
    """Pick the machine of a given microarchitecture to run on.

    Reuses g5k-availability's view of the testbed and takes the node that is
    free the soonest. Unless ``wait`` is set we insist on a node that is free
    *now*, so that a mistyped microarchitecture or a busy testbed fails here
    rather than parking a reservation hours into the future.
    """
    chosen_sites = list(sites or sorted(site.uid for site in client().sites.list()))
    chosen_sites = [site for site in chosen_sites if site not in set(exclude_sites)]
    machines = discover(microarch, chosen_sites, workers)
    if exclude_nodes:
        unwanted = {normalize_server(node) for node in exclude_nodes}
        machines = [machine for machine in machines if machine.fqdn not in unwanted]
    if not machines:
        raise SystemExit(
            f"no machine with a {microarch!r} microarchitecture across "
            f"{len(chosen_sites)} site(s)"
        )

    now = int(time.time())
    annotate_availability(machines, walltime, now, workers)
    reservable = [m for m in machines if m.available_at is not None]
    if not reservable:
        raise SystemExit(
            f"none of the {len(machines)} {microarch!r} machine(s) can host a "
            f"{walltime // 3600}h job (all dead, absent or walltime-capped)"
        )

    reservable.sort(key=sort_key)
    free_now = [m for m in reservable if (m.available_at or 0) <= now]
    if not free_now and not wait:
        soonest = reservable[0]
        raise SystemExit(
            f"no {microarch!r} machine is free right now; soonest is "
            f"{soonest.uid} on {soonest.site}, free {format_when(soonest, now)}. "
            "Pass --wait to queue the job anyway, or --reservation to book it."
        )

    # A standby node is free but has to be booted first, which costs minutes
    # at the start of the job: prefer one that is already up.
    free_now.sort(key=lambda machine: machine.hard_state != "alive")
    chosen = (free_now or reservable)[0]
    logger.info(
        "%d %r machine(s), %d free now; taking %s (%s, %s cores, queue=%s)",
        len(machines),
        microarch,
        len(free_now),
        chosen.fqdn,
        chosen.cpu,
        chosen.cores,
        chosen.queue,
    )
    return chosen


def resolve_target(args: argparse.Namespace, walltime: int) -> Target:
    """Turn the selection flags into a single node to reserve."""
    if args.server:
        server = normalize_server(args.server)
        site = server.split(".")[1]
        cluster = server.split(".")[0].rsplit("-", 1)[0]
        return Target(
            site=site,
            queue=args.queue or cluster_queue(site, cluster),
            server=server,
            description=server,
        )
    if args.cluster:
        site = get_cluster_site(args.cluster)
        return Target(
            site=site,
            queue=args.queue or cluster_queue(site, args.cluster),
            cluster=args.cluster,
            description=f"any node of {args.cluster} ({site})",
        )
    machine = pick_machine(
        args.microarch,
        args.sites,
        args.exclude_sites,
        args.exclude_nodes,
        walltime,
        args.workers,
        args.wait or bool(args.reservation),
    )
    return Target(
        site=machine.site,
        queue=args.queue or machine.queue,
        server=machine.fqdn,
        description=f"{machine.fqdn} [{machine.microarchitecture}, {machine.cpu}]",
        verified=True,
    )


def node_description(site: str, fqdn: str) -> dict | None:
    """The reference API's full description of one node.

    Worth keeping beside the results: it is the independent account of the
    hardware -- exact CPU stepping, cache sizes, DIMM layout, BIOS version --
    against which whatever the job detected from inside a container can be
    checked. Best effort; a run is not worth failing over its absence.
    """
    uid = fqdn.split(".")[0]
    cluster = uid.rsplit("-", 1)[0]
    try:
        node = client().sites[site].clusters[cluster].nodes[uid]
        return node.to_dict()
    except Exception as error:
        logger.debug("no reference API description for %s: %s", fqdn, error)
        return None


def site_nodes(site: str) -> dict:
    """OAR's view of every node of a site, reservations included."""
    statuses = client().sites[site].status
    try:
        # `waiting=yes` also reports jobs OAR has queued but not started, which
        # is what makes a reservation in the near future visible here.
        status = statuses.list(waiting="yes")
    except Exception:
        status = statuses.list()
    return getattr(status, "nodes", None) or {}


def verify_free(target: Target, walltime: int, wait: bool) -> None:
    """Refuse to reserve a node that is not free yet, unless asked to queue.

    OAR happily accepts a job for a busy node and starts it hours later, and
    enoslib then waits for it without a timeout, which looks exactly like a
    hang. Checking first turns that into an answer.
    """
    if target.verified:
        return
    nodes = site_nodes(target.site)
    if target.server:
        candidates = {target.server: nodes.get(target.server)}
    else:
        prefix = f"{target.cluster}-"
        candidates = {
            fqdn: status
            for fqdn, status in nodes.items()
            if fqdn.split(".")[0].startswith(prefix)
        }
    now = int(time.time())
    free = {
        fqdn: soonest_start(status.get("reservations") or [], now, walltime)
        for fqdn, status in candidates.items()
        if status is not None and status.get("hard") in RESERVABLE_STATES
    }
    if not free:
        raise SystemExit(
            f"{target.description} cannot be reserved: no node of it is alive "
            "in OAR (dead, absent, or retired)"
        )
    soonest = min(free.values())
    if soonest > now and not wait:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(soonest))
        raise SystemExit(
            f"{target.description} is busy; the next {walltime // 3600}h slot "
            f"starts {stamp}. Pass --wait to queue the job anyway, or "
            "--reservation to book it."
        )


# OAR's own account of a job is the only thing that says *why* it failed; the
# exception enoslib raises just carries the job's JSON.
OAR_NOISE = ("ADM_RULES_MSG_OUT",)


def job_events(site: str, job_name: str, login: str) -> list[str]:
    """OAR's events for our most recent finished job of that name, newest last."""
    try:
        jobs = (
            client()
            .sites[site]
            .jobs.list(name=job_name, state="error,terminated", user=login)
        )
    except Exception:
        return []
    if not jobs:
        return []
    latest = max(jobs, key=lambda job: getattr(job, "submitted_at", 0) or 0)
    try:
        latest.refresh()
    except Exception:
        pass
    events = getattr(latest, "events", None) or []
    return [
        f"[{event.get('type')}] "
        f"{(event.get('description') or '').splitlines()[0].strip()}"
        for event in events
        if event.get("type") not in OAR_NOISE
    ]


def describe_reservation_failure(
    site: str, job_name: str, login: str, error: Exception
) -> str:
    """Turn enoslib's job dump into the couple of lines that matter."""
    lines = [f"error: the reservation for {job_name} on {site} did not start"]
    events = job_events(site, job_name, login)
    if events:
        lines += [f"  {event}" for event in events[-3:]]
        if any("working directory" in event.lower() for event in events):
            # OAR refuses to start a job whose submission directory it cannot
            # reach, which on a node means the NFS home is not usable there.
            lines.append(
                "hint: the node cannot reach your home directory, so nothing "
                "can run on it. That is the node being broken rather than the "
                "job: pick another one (--exclude-nodes or --server), and it "
                "is worth reporting to Grid'5000 support."
            )
    else:
        lines.append(f"  {str(error).splitlines()[0][:200]}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The script that runs on the node
# --------------------------------------------------------------------------- #


@dataclass
class Layout:
    """Where a run's files live on the site's NFS home."""

    home: str
    # Home-relative, so the same path also works over access.grid5000.fr.
    relative: PurePosixPath
    run_id: str

    @property
    def rundir(self) -> str:
        return f"{self.home}/{self.relative}"

    @property
    def log(self) -> str:
        return f"{self.rundir}/run.log"

    @property
    def exit_code(self) -> str:
        return f"{self.rundir}/exit_code"

    @property
    def results(self) -> str:
        return f"{self.rundir}/results"


def make_layout(login: str, remote_dir: str, run_id: str) -> Layout:
    return Layout(
        home=f"/home/{login}",
        relative=PurePosixPath(remote_dir) / run_id,
        run_id=run_id,
    )


def render_script(
    *,
    layout: Layout,
    image: str,
    image_tar: str | None,
    pull: bool,
    docker_args: Sequence[str],
    command: Sequence[str],
    node_setup: Sequence[str] = (),
) -> str:
    """Build the bash script the node runs, detached, to produce the run.

    It is deliberately one self-contained file: it is written into the run
    directory, so what ran is recoverable afterwards, and nothing depends on
    the SSH connection staying up for the container to keep going.
    """
    # Where the container is running. A container's own hostname is its id, so
    # without this nothing inside it can tell which node, cluster or site it
    # landed on -- which is exactly what a command that configures itself from
    # the hardware needs to know. Expanded on the node, not here, because with
    # --cluster the node is only settled once OAR has picked one.
    location = [
        "-e",
        'G5K_NODE="$node"',
        "-e",
        'G5K_CLUSTER="$cluster"',
        "-e",
        'G5K_SITE="$site"',
        "-e",
        'G5K_JOB_ID="${OAR_JOB_ID:-}"',
    ]
    docker_run = " ".join(
        ["docker", "run", "--rm", "--name", shlex.quote(layout.run_id)]
        + location
        + [shlex.quote(arg) for arg in docker_args]
        + ['"$image"']
        + [shlex.quote(word) for word in command]
    )

    # Things that have to happen on the node rather than in the container:
    # writing a sysctl that is not namespaced, fixing the CPU clock, staging
    # data on node-local disk. They go in before the image is fetched, so a
    # setup that fails does not first cost a multi-gigabyte pull, and they land
    # in the archived run.sh, so the run records how the node was prepared.
    setup = ""
    if node_setup:
        lines = ["", "# --node-setup, in the order given."]
        for entry in node_setup:
            lines.append(f"echo {shlex.quote('[g5k-run] setup: ' + entry)}")
            lines.append(entry)
        setup = "\n".join(lines) + "\n"

    if image_tar:
        # `docker load` prints the tag it restored, which is how we can run a
        # tarball whose image name the caller did not spell out.
        acquire = f"""  echo "[g5k-run] loading {image_tar}"
  loaded=$(docker load -i {shlex.quote(image_tar)})
  printf '%s\\n' "$loaded"
  if [ -z "$image" ]; then
    image=$(printf '%s\\n' "$loaded" | sed -n 's/^Loaded image: //p' | head -n1)
  fi"""
    elif pull:
        acquire = """  echo "[g5k-run] pulling $image"
  docker pull "$image\""""
    else:
        acquire = """  echo "[g5k-run] image $image is not on this node; pass --pull or --image-tar" >&2
  exit 2"""

    return f"""#!/usr/bin/env bash
# Generated by g5k-run; rewritten on every run.
set -o pipefail
# Always leave the exit code behind: the poller waits for this file, and a
# failure in the setup steps has to end the wait just like the container does.
trap 'rc=$?; printf "%s\\n" "$rc" > {layout.exit_code}' EXIT
set -e

node=$(hostname -f)
# grdix-5.nancy.grid5000.fr -> grdix, nancy. Grid'5000 numbers the nodes of a
# cluster, and a cluster's nodes are identical, so the cluster is the useful
# name for the hardware.
short=$(hostname -s)
cluster=${{short%-*}}
site=$(printf '%s' "$node" | cut -d. -f2)

echo "[g5k-run] $node, $(date -Is), job ${{OAR_JOB_ID:-none}}"

if command -v docker >/dev/null 2>&1; then
  echo "[g5k-run] docker already installed"
else
  echo "[g5k-run] installing docker (g5k-setup-docker -t, images under /tmp)"
  g5k-setup-docker -t
fi
{setup}
image={shlex.quote(image)}
if [ -n "$image" ] && docker image inspect "$image" >/dev/null 2>&1; then
  echo "[g5k-run] image $image already on the node"
else
{acquire}
fi
if [ -z "$image" ]; then
  echo "[g5k-run] could not determine the image to run; pass --image" >&2
  exit 2
fi

echo "[g5k-run] image $image"
# A quoted heredoc so the command reaches the log exactly as it runs, however
# many quotes and dollars the caller put in it.
cat <<'G5KRUN_COMMAND'
[g5k-run] {docker_run}
G5KRUN_COMMAND
{docker_run}
"""


# --------------------------------------------------------------------------- #
# Driving the node
# --------------------------------------------------------------------------- #


def as_user(host: en.Host, login: str) -> en.Host:
    """The same host, but connected to as the Grid'5000 user.

    enoslib hands out hosts as root (it runs sudo-g5k while reserving), which
    would put us in /root with the NFS home squashed away. `g5k-setup-docker`
    escalates on its own, so the user account is both enough and the only one
    whose `$HOME` is the shared home directory.
    """
    return en.Host(address=host.address, user=login, extra=host.get_extra())


def remote(
    host: en.Host, command: str, *, task: str, allow_failure: bool = False
) -> str:
    """Run one command on the node over SSH and return its stdout.

    `raw` keeps this a plain SSH command: no Python needed on the node, and no
    Ansible module round trip for what are one-liners.
    """
    try:
        results = en.run_command(
            command,
            roles=[host],
            raw=True,
            task_name=task,
            on_error_continue=allow_failure,
        )
    except Exception as error:
        if allow_failure:
            return ""
        raise SystemExit(f"error: {task} failed on {host.address}: {error}") from None
    result = results[0] if results else None
    if result is None or not result.ok():
        if allow_failure:
            return ""
        detail = getattr(result, "stderr", None) or getattr(result, "status", "?")
        raise SystemExit(f"error: {task} failed on {host.address}: {detail}")
    return result.stdout or ""


def upload(host: en.Host, path: str, content: str, *, task: str) -> None:
    """Write a file on the node without depending on quoting or on Ansible.

    The content goes over base64-encoded: Ansible templates the command it
    sends, so a `{{` in a user command would otherwise be eaten, and the shell
    would take its own bite out of the rest.
    """
    payload = base64.b64encode(content.encode()).decode()
    remote(
        host,
        f"printf '%s' {shlex.quote(payload)} | base64 -d > {shlex.quote(path)}",
        task=task,
    )


def launch(host: en.Host, layout: Layout) -> None:
    """Start the run script, detached from our SSH session, and check it ran.

    Ansible's own detached mode does the daemonising: a hand-rolled
    ``nohup ... &`` races sshd tearing the session's process group down, and
    when it loses that race the SSH command still reports success. The log
    file appearing is the proof that the script actually started.
    """
    en.run_command(
        f"bash {shlex.quote(layout.rundir)}/run.sh > {shlex.quote(layout.log)} 2>&1",
        roles=[host],
        background=True,
        task_name="Starting the run",
    )
    for _ in range(10):
        started = remote(
            host,
            f"test -f {shlex.quote(layout.log)} && echo yes || echo no",
            task="Checking the run started",
            allow_failure=True,
        )
        if started.strip().endswith("yes"):
            return
        time.sleep(2.0)
    raise SystemExit(
        f"error: the run did not start: nothing created {layout.log} on "
        f"{host.address}. Look for ~/.ansible_async/ on the node."
    )


def parse_polled(stdout: str) -> tuple[bytes, str | None]:
    """Split a poll's output into new log bytes and the exit code, if any.

    The exit code only counts once its trailing newline is there: a poll can
    always land between the two writes that produce it.
    """
    match = POLLED.search(stdout)
    if match is None:
        return b"", None
    chunk = base64.b64decode(match.group(1)) if match.group(1) else b""
    written = match.group(2)
    code = written.strip() if written.endswith("\n") else ""
    return chunk, code or None


def stream(host: en.Host, layout: Layout, interval: float, provider) -> int:
    """Follow the run's log until it ends, and return the container's exit code.

    The log is a file on the NFS home rather than our SSH session's stdout, so
    the run survives a lost connection and stays readable from the frontend
    (`tail -f`) while it goes.
    """
    offset = 0

    def poll(allow_failure: bool) -> tuple[bytes, str | None, bool]:
        nonlocal offset
        with DisableLogging(level=logging.ERROR if allow_failure else logging.NOTSET):
            stdout = remote(
                host,
                f"printf '@@LOG@@'; tail -c +{offset + 1} {shlex.quote(layout.log)} "
                f"2>/dev/null | base64 -w0; printf '@@RC@@'; "
                f"cat {shlex.quote(layout.exit_code)} 2>/dev/null; printf '@@END@@'",
                task="Reading the log",
                allow_failure=allow_failure,
            )
        if not stdout:
            return b"", None, False
        chunk, code = parse_polled(stdout)
        offset += len(chunk)
        return chunk, code, True

    def emit(chunk: bytes) -> None:
        if chunk:
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()

    failures = 0
    warned = False
    while True:
        chunk, code, reached = poll(allow_failure=True)
        emit(chunk)
        if not reached:
            failures += 1
            if failures >= 3:
                # Three misses in a row is not a hiccup. Either OAR took the
                # job back, or the node went quiet on us -- which does not stop
                # the run itself: it is detached and writes to the NFS home.
                states = [job.state for job in provider.jobs]
                if not all(state == "running" for state in states):
                    raise SystemExit(
                        f"error: the job is no longer running (state {states}); "
                        "the walltime was probably reached"
                    )
                if not warned:
                    warned = True
                    logger.warning(
                        "%s has stopped answering, retrying every %gs. The run "
                        "keeps going; `tail -f %s` from the site frontend also "
                        "shows it, and the results are fetched either way",
                        host.address,
                        interval,
                        layout.log,
                    )
                failures = 0
            time.sleep(interval)
            continue
        if warned:
            warned = False
            logger.info("%s is answering again", host.address)
        failures = 0
        if code is not None:
            # The exit code is written after the last line of output, so one
            # more read is needed to catch up with it.
            chunk, _, _ = poll(allow_failure=True)
            emit(chunk)
            return int(code) if code.isdigit() else 1
        time.sleep(interval)


# --------------------------------------------------------------------------- #
# Moving files in and out
# --------------------------------------------------------------------------- #


def ssh_access(login: str, command: str) -> subprocess.CompletedProcess:
    """Run a one-liner on the access machine.

    `BatchMode` keeps a probe from sitting on a passphrase prompt it cannot
    show; the transfers that follow use a normal SSH and can still ask.
    """
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", f"{login}@{ACCESS}", command],
        capture_output=True,
        text=True,
        check=False,
    )


def check_ssh_setup(login: str, site: str) -> None:
    """Check the two SSH things whose absence only shows up after reserving.

    Both are worth a round trip before taking a node: everything we do on the
    node goes through ``<login>@access.grid5000.fr`` (enoslib jumps over it,
    and so do our transfers), and enoslib grants itself root while reserving
    with ``cat ~/.ssh/id_rsa.pub ~/.ssh/authorized_keys | sudo-g5k ...``, so
    both files have to exist in the site's home even though we only ever
    connect as the user.
    """
    probe = ssh_access(
        login, f"test -r {site}/.ssh/id_rsa.pub && test -r {site}/.ssh/authorized_keys"
    )
    if probe.returncode == 0:
        return
    if probe.returncode == 255:
        lines = (probe.stderr or "").strip().splitlines()
        raise SystemExit(
            f"error: cannot ssh to {login}@{ACCESS} non-interactively "
            f"({lines[-1] if lines else 'no output'}).\n"
            f"hint: everything here goes through that host, including enoslib's "
            f"jump to the node, so nothing would work once the node is "
            f"reserved. Run `ssh {login}@{ACCESS} hostname` once by hand to "
            f"sort the host key or the agent out, then retry."
        )
    logger.warning(
        "no readable ~/.ssh/id_rsa.pub and ~/.ssh/authorized_keys in your %s "
        "home; enoslib needs both to reserve a node (run `ssh-keygen -t rsa` "
        "on the frontend, then append the key to authorized_keys)",
        site,
    )


def push_image(
    local: Path, login: str, site: str, relative: PurePosixPath, force: bool
) -> None:
    """Copy an image tarball to the site's NFS home, unless it is there already.

    Home directories are per site and shared with that site's nodes, which is
    what makes `docker load` on the node cheap and keeps us off Docker Hub (and
    out of its rate limits) entirely. Multi-gigabyte tarballs are the norm here,
    so a matching size is taken as "same file".
    """
    if not local.is_file():
        raise SystemExit(f"error: {local} is not a file")
    remote_path = f"{site}/{relative}"
    probe = ssh_access(login, f"stat -c %s {shlex.quote(remote_path)}")
    remote_size = int(probe.stdout.strip()) if probe.returncode == 0 else None
    if remote_size == local.stat().st_size and not force:
        logger.info("%s is already on %s (%d bytes)", relative, site, remote_size)
        return

    logger.info(
        "copying %s (%.1f GiB) to %s:%s",
        local,
        local.stat().st_size / 1024**3,
        site,
        relative,
    )
    parent = str(PurePosixPath(remote_path).parent)
    ssh_access(login, f"mkdir -p {shlex.quote(parent)}")
    subprocess.run(
        [
            "rsync",
            "-a",
            "--info=progress2",
            str(local),
            f"{login}@{ACCESS}:{remote_path}",
        ],
        check=True,
    )


def fetch(login: str, site: str, layout: Layout, destination: Path) -> Path | None:
    """Bring the run directory (results, log, script) back here."""
    destination.mkdir(parents=True, exist_ok=True)
    local = destination / layout.run_id
    source = f"{login}@{ACCESS}:{site}/{layout.relative}/"
    logger.info("fetching %s into %s", source, local)
    completed = subprocess.run(
        ["rsync", "-az", source, f"{local}/"],
        check=False,
    )
    if completed.returncode != 0:
        logger.warning(
            "rsync failed; the run is still on %s, fetch it with:\n  rsync -az %s %s/",
            site,
            source,
            local,
        )
        return None
    return local


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reserve one Grid'5000 node and run a command in a Docker "
            "container on it, streaming the output and fetching the results."
        ),
        epilog="The command to run in the container goes after a -- separator.",
    )

    selection = parser.add_argument_group("node selection")
    exclusive = selection.add_mutually_exclusive_group()
    exclusive.add_argument(
        "--server",
        metavar="NODE",
        help="reserve this exact node, e.g. gres-1.nancy.grid5000.fr",
    )
    exclusive.add_argument(
        "--cluster", metavar="NAME", help="reserve any node of this cluster"
    )
    exclusive.add_argument(
        "--microarch",
        default="zen 4",
        metavar="NAME",
        help=(
            "reserve the soonest free node with this microarchitecture "
            "(default: %(default)r); see g5k-availability"
        ),
    )
    selection.add_argument(
        "--sites", nargs="+", metavar="SITE", help="only search these sites"
    )
    selection.add_argument(
        "--exclude-sites",
        nargs="+",
        default=[],
        metavar="SITE",
        help="skip these sites",
    )
    selection.add_argument(
        "--exclude-nodes",
        nargs="+",
        default=[],
        metavar="NODE",
        help="never pick these nodes, e.g. one whose home mount is broken",
    )
    selection.add_argument(
        "--queue", help="OAR queue (default: the target's own, e.g. production)"
    )
    selection.add_argument(
        "--wait",
        action="store_true",
        help="queue the job even if no matching node is free right now",
    )

    job = parser.add_argument_group("reservation")
    job.add_argument(
        "--walltime", default="1:00:00", help="job walltime (default: %(default)s)"
    )
    job.add_argument(
        "--job-name",
        help=(
            "OAR job name (default: g5k-run-<target>). A job of this name is "
            "reloaded rather than duplicated, so re-runs reuse the same node."
        ),
    )
    job.add_argument(
        "--reservation",
        metavar="'YYYY-MM-DD HH:MM:SS'",
        help="book the node for this date instead of as soon as possible",
    )
    job.add_argument(
        "--keep",
        action="store_true",
        help=(
            "leave the job running afterwards (Docker and the image stay warm "
            "for the next run; the node stays reserved until the walltime)"
        ),
    )

    image = parser.add_argument_group("image")
    image.add_argument(
        "--image",
        default="",
        help="image to run, e.g. ghcr.io/opencompl/xdsl-autotuning-ci:0.31.0",
    )
    image.add_argument(
        "--pull", action="store_true", help="pull --image on the node if it is missing"
    )
    image.add_argument(
        "--image-tar",
        metavar="PATH",
        help=(
            "local `docker save` tarball to copy to the site's home and load "
            "on the node (skipped when the copy is already there)"
        ),
    )
    image.add_argument(
        "--remote-image-tar",
        metavar="PATH",
        help="tarball already on the site's home, relative to it or absolute",
    )
    image.add_argument(
        "--force-push", action="store_true", help="copy the tarball even if it is there"
    )

    node = parser.add_argument_group("node")
    node.add_argument(
        "--node-setup",
        action="append",
        default=[],
        metavar="CMD",
        help=(
            "shell command to run on the node, as your user, before the "
            "container starts; repeatable and run in order. For what the "
            "container cannot do itself: `sudo sysctl -w "
            "kernel.perf_event_paranoid=-1` (not namespaced, so it has to "
            "happen here), fixing the CPU clock, or staging data on the "
            "node's local disk."
        ),
    )

    container = parser.add_argument_group("container")
    container.add_argument(
        "--mount",
        action="append",
        default=[],
        metavar="SRC:DST",
        help="bind mount, repeatable; <rundir>/results:/results is always set",
    )
    container.add_argument(
        "--workdir", metavar="PATH", help="container working directory"
    )
    container.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="K=V",
        help="environment variable for the container, repeatable",
    )
    container.add_argument(
        "--docker-arg",
        action="append",
        default=[],
        metavar="ARG",
        help=(
            "extra `docker run` argument, repeatable. Write it attached, as "
            "--docker-arg=--cap-add=PERFMON, so argparse does not read it as "
            "an option of ours."
        ),
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--remote-dir",
        default="g5k-run",
        help="directory on the site's home holding the runs (default: %(default)s)",
    )
    output.add_argument(
        "--fetch-to",
        default="g5k-runs",
        metavar="DIR",
        help="local directory to rsync the run into (default: %(default)s)",
    )
    output.add_argument(
        "--no-fetch", action="store_true", help="leave the results on Grid'5000"
    )
    output.add_argument(
        "--poll",
        type=float,
        default=15.0,
        metavar="SECONDS",
        help="how often to read the remote log (default: %(default)s)",
    )
    output.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the node script, reserve nothing",
    )
    output.add_argument("--verbose", action="store_true", help="debug logging")
    output.add_argument(
        "--workers",
        type=int,
        default=8,
        help="parallel API requests when searching (default: %(default)s)",
    )

    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        metavar="-- COMMAND [ARG ...]",
        help="the command to run inside the container",
    )
    return parser.parse_args(argv)


def container_command(argv: Sequence[str]) -> list[str]:
    """The words after the `--` separator argparse leaves in place."""
    words = list(argv)
    if words and words[0] == "--":
        words = words[1:]
    return words


def docker_arguments(args: argparse.Namespace, layout: Layout) -> list[str]:
    """Assemble the `docker run` arguments, results mount first."""
    arguments = ["-v", f"{layout.results}:/results"]
    for mount in args.mount:
        arguments += ["-v", mount]
    for variable in args.env:
        arguments += ["-e", variable]
    if args.workdir:
        arguments += ["-w", args.workdir]
    arguments += args.docker_arg
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    # Our streamed container output is the interesting thing on stdout; keep
    # Ansible's own chatter out of it.
    en.set_config(ansible_stdout="noop")

    command = container_command(args.command)
    if not command:
        return fail("nothing to run: put the container command after a -- separator")
    try:
        walltime = parse_walltime(args.walltime)
    except ValueError as error:
        return fail(str(error))
    if args.image_tar and args.remote_image_tar:
        return fail("--image-tar and --remote-image-tar are mutually exclusive")
    if args.pull and not args.image:
        return fail("--pull needs --image to say what to pull")
    if not args.image and not (args.image_tar or args.remote_image_tar):
        return fail("pass --image (with --pull) or --image-tar")

    try:
        login = get_api_username()
    except Exception as error:
        return fail(f"cannot read your Grid'5000 login ({error})")
    if not login:
        return fail(
            "no Grid'5000 login found; set it in ~/.python-grid5000.yaml or in G5K_USER"
        )

    try:
        target = resolve_target(args, walltime)
        verify_free(target, walltime, args.wait or bool(args.reservation))
    except (ValueError, SystemExit) as error:
        return fail(str(error))

    job_name = args.job_name or f"g5k-run-{target.label}"
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{target.label}"
    layout = make_layout(login, args.remote_dir, run_id)

    # A home-relative tarball path is what makes the same string usable from
    # the node and from access.grid5000.fr.
    if args.image_tar:
        tar_relative = (
            PurePosixPath(args.remote_dir) / "images" / Path(args.image_tar).name
        )
        tar_on_node = f"{layout.home}/{tar_relative}"
    elif args.remote_image_tar:
        tar_relative = None
        tar_on_node = args.remote_image_tar
        if not tar_on_node.startswith("/"):
            tar_on_node = f"{layout.home}/{tar_on_node}"
    else:
        tar_relative = None
        tar_on_node = None

    script = render_script(
        layout=layout,
        image=args.image,
        image_tar=tar_on_node,
        pull=args.pull,
        docker_args=docker_arguments(args, layout),
        command=command,
        node_setup=args.node_setup,
    )
    metadata = {
        "run_id": run_id,
        "job_name": job_name,
        "site": target.site,
        "target": target.description,
        "queue": target.queue,
        "walltime": args.walltime,
        "image": args.image or None,
        "image_tar": tar_on_node,
        "command": command,
        "node_setup": list(args.node_setup),
        "submitted_at": int(time.time()),
    }

    print(f"target:   {target.description}")
    print(f"site:     {target.site}  queue: {target.queue}  walltime: {args.walltime}")
    print(f"job name: {job_name} (reloaded if it already exists)")
    print(f"run dir:  {target.site}:{layout.relative}")
    print(f"results:  {layout.results} -> /results in the container")
    if args.dry_run:
        print("\n--- run.sh ------------------------------------------------")
        print(script, end="")
        print("-----------------------------------------------------------")
        print("\ndry run: nothing was reserved")
        return 0

    try:
        check_ssh_setup(login, target.site)
    except SystemExit as error:
        return fail(str(error).removeprefix("error: "))

    if args.image_tar and tar_relative is not None:
        push_image(
            Path(args.image_tar).expanduser(),
            login,
            target.site,
            tar_relative,
            args.force_push,
        )

    existing = grid_reload_jobs_from_name(job_name, restrict_to=[target.site])
    if existing:
        logger.info(
            "reusing job(s) %s on %s",
            ", ".join(f"{job.uid} ({job.state})" for job in existing),
            target.site,
        )

    conf = en.G5kConf.from_settings(
        job_name=job_name,
        job_type=[],
        walltime=args.walltime,
        queue=target.queue,
        reservation=args.reservation,
    )
    if target.server:
        conf.add_machine(roles=[ROLE], servers=[target.server])
    else:
        conf.add_machine(roles=[ROLE], cluster=target.cluster, nodes=1)
    provider = en.G5k(conf.finalize())

    exit_code = 1
    host: en.Host | None = None
    try:
        try:
            roles, _ = provider.init()
        except Exception as error:
            raise SystemExit(
                describe_reservation_failure(target.site, job_name, login, error)
            ) from None
        host = as_user(roles[ROLE][0], login)
        job_ids = ", ".join(str(job.uid) for job in provider.jobs)
        logger.info("job %s running on %s", job_ids, host.address)
        # A node OAR just woke from standby, or one still rebooting after an
        # earlier sudo-g5k job, can accept a connection before it has settled.
        logger.info("waiting for %s to answer", host.address)
        with DisableLogging(level=logging.ERROR):
            en.wait_for([host], retries=20, interval=15)
        metadata["job_ids"] = job_ids
        metadata["node"] = host.address

        remote(
            host,
            f"mkdir -p {shlex.quote(layout.results)}",
            task="Creating the run directory",
        )
        upload(host, f"{layout.rundir}/run.sh", script, task="Uploading run.sh")
        upload(
            host,
            f"{layout.rundir}/meta.json",
            json.dumps(metadata, indent=2) + "\n",
            task="Uploading meta.json",
        )
        if (description := node_description(target.site, host.address)) is not None:
            upload(
                host,
                f"{layout.rundir}/node.json",
                json.dumps(description, indent=2, sort_keys=True) + "\n",
                task="Uploading node.json",
            )
        launch(host, layout)
        print(
            f"--- output ({login}@{target.site}: tail -f {layout.log}) ---",
            flush=True,
        )
        exit_code = stream(host, layout, args.poll, provider)
        print(f"--- container exited with {exit_code} ---")
    except KeyboardInterrupt:
        stage = "stopping the container" if host is not None else "giving the node back"
        print(f"\ninterrupted; {stage}", file=sys.stderr)
        if host is not None:
            remote(
                host,
                f"docker rm -f {shlex.quote(run_id)} >/dev/null 2>&1 || true",
                task="Stopping the container",
                allow_failure=True,
            )
        exit_code = 130
    finally:
        if args.keep:
            logger.info(
                "keeping the job; delete it with "
                "`oardel <id>` on %s or rerun with the same --job-name",
                target.site,
            )
        else:
            provider.destroy()

    if not args.no_fetch:
        local = fetch(login, target.site, layout, Path(args.fetch_to).expanduser())
        if local is not None:
            print(f"run directory: {local}")
    return exit_code


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
