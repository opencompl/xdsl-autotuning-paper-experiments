"""List Grid'5000 machines of a given microarchitecture and their soonest availability.

Walks every Grid'5000 site through the reference API, keeps the nodes whose
processor microarchitecture matches (Zen 4 by default, which also covers the
denser Zen 4c parts), then reads each site's OAR status to work out the earliest
moment a node is free for an uninterrupted `--walltime` window.

    cd g5k-tools
    uv run g5k-availability
    uv run g5k-availability --microarch "zen 5" --walltime 4:00:00
    uv run g5k-availability --json

or, independent of the environment, g5k-tools/.venv/bin/g5k-availability.

This lives in its own uv project because enoslib pins `rich ~= 12.0`, which the
main venv's xdsl[dev] cannot accept; see g5k-tools/README.md.

Needs Grid'5000 API credentials, either in ~/.python-grid5000.yaml

    username: <login>
    password: <password>

or via the G5K_USER / G5K_PASSWORD environment variables. On a Grid'5000
frontend the anonymous connection is enough.
"""

import argparse
from concurrent import futures
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable, Sequence
from typing import Any

from enoslib.infra.enos_g5k.g5k_api_utils import Client

CONFIG_FILE = Path.home() / ".python-grid5000.yaml"

# OAR hands out besteffort jobs on the understanding that they get killed as
# soon as a regular job wants the node, so they never delay an availability.
BESTEFFORT_QUEUE = "besteffort"

# A node whose `hard` state is none of these cannot be reserved at all,
# whatever its reservation calendar says. `standby` is a node Grid'5000 powered
# down to save energy: it is idle and reservable, OAR just has to boot it
# first, which costs a couple of minutes at the start of the job.
STANDBY = "standby"
RESERVABLE_STATES = ("alive", STANDBY)


# --------------------------------------------------------------------------- #
# API access
# --------------------------------------------------------------------------- #

_local = threading.local()


def make_client() -> Client:
    """Build a python-grid5000 client from the environment or the config file."""
    user, password = os.environ.get("G5K_USER"), os.environ.get("G5K_PASSWORD")
    if user and password:
        return Client(username=user, password=password)
    if CONFIG_FILE.exists():
        # Client.from_yaml silently falls back to an anonymous connection when
        # the file is unreadable, which is the right behaviour on a frontend.
        return Client.from_yaml(str(CONFIG_FILE))
    return Client()


def client() -> Client:
    """Return this thread's client.

    python-grid5000 wraps a requests.Session, which is not meant to be shared
    across threads, so each worker gets its own rather than reusing enoslib's
    process-wide singleton.
    """
    existing = getattr(_local, "client", None)
    if existing is None:
        existing = _local.client = make_client()
    return existing


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #


def parse_walltime(walltime: str) -> int:
    """Turn an OAR walltime (``H:MM:SS``, ``H:MM`` or ``H``) into seconds."""
    parts = walltime.split(":")
    if len(parts) > 3:
        raise ValueError(f"invalid walltime {walltime!r}")
    try:
        fields = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"invalid walltime {walltime!r}") from None
    while len(fields) < 3:
        fields.append(0)
    hours, minutes, seconds = fields
    total = hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise ValueError(f"walltime {walltime!r} must be positive")
    return total


def busy_intervals(reservations: Iterable[dict[str, Any]]) -> list[tuple[int, int]]:
    """Return the sorted ``(start, end)`` windows during which a node is taken."""
    intervals = []
    for reservation in reservations:
        if reservation.get("queue") == BESTEFFORT_QUEUE:
            continue
        start = reservation.get("started_at") or reservation.get("scheduled_at")
        walltime = reservation.get("walltime")
        if start is None or walltime is None:
            # A job OAR has not scheduled yet: we cannot place it on the
            # timeline, so it cannot constrain the answer either.
            continue
        start = int(start)
        intervals.append((start, start + int(walltime)))
    intervals.sort()
    return intervals


def soonest_start(
    reservations: Iterable[dict[str, Any]], now: int, walltime: int
) -> int:
    """Earliest timestamp at which ``walltime`` seconds are free in a row.

    Reservations are sorted by start date, so we sweep them once and push the
    candidate start past every window that would overlap it.
    """
    candidate = now
    for start, end in busy_intervals(reservations):
        if end <= candidate:
            # Already behind the candidate window.
            continue
        if start >= candidate + walltime:
            # The gap in front of this reservation is wide enough; because the
            # windows are sorted, nothing later can shrink it.
            break
        candidate = end
    return candidate


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


@dataclass
class Machine:
    site: str
    cluster: str
    uid: str
    microarchitecture: str
    cpu: str
    cores: int
    threads: int
    memory_gib: float
    gpus: int
    exotic: bool
    sockets: int = 1
    queues: list[str] = field(default_factory=list)
    max_walltime: int | None = None
    hard_state: str = "unknown"
    soft_state: str = "unknown"
    available_at: int | None = None
    blocked_by: str | None = None
    reservations: int = 0
    besteffort: int = 0

    @property
    def fqdn(self) -> str:
        return f"{self.uid}.{self.site}.grid5000.fr"

    @property
    def queue(self) -> str:
        """The OAR queue to reserve this node through."""
        return preferred_queue(self.queues)


def preferred_queue(queues: Sequence[str]) -> str:
    """The OAR queue to reserve a node through, given the queues it accepts.

    Production nodes are not in the default queue, so a reservation needs an
    explicit ``-q production``; surfacing this avoids a puzzling refusal.
    """
    for candidate in ("default", "production", "testing"):
        if candidate in queues:
            return candidate
    return queues[0] if queues else "unknown"


def matches(microarchitecture: str | None, wanted: str) -> bool:
    """Match a reference-API microarchitecture against the requested one.

    The reference API spells these with a space (``"Zen 4"``, ``"Zen 4c"``,
    ``"Skylake-SP"``), so we compare case-insensitively on a prefix: asking for
    ``zen 4`` finds both Zen 4 and Zen 4c, while ``zen 4c`` narrows it down.
    """
    if not microarchitecture:
        return False
    return microarchitecture.strip().casefold().startswith(wanted.strip().casefold())


def describe_cpu(processor: dict[str, Any], architecture: dict[str, Any]) -> str:
    """Name the CPU, made explicit about socket count.

    The reference API's ``other_description`` names a single package (e.g. "AMD
    EPYC 9254 24-Core Processor"), while ``nb_cores`` counts the whole node, so
    a two-socket node otherwise looks like it reports twice its CPU's cores.
    """
    name = processor.get("other_description") or " ".join(
        str(part) for part in (processor.get("model"), processor.get("version")) if part
    )
    sockets = architecture.get("nb_procs") or 1
    return f"{sockets}x {name}" if sockets > 1 else name


def cluster_machines(site: str, cluster: str, wanted: str) -> list[Machine]:
    """Return the matching nodes of one cluster."""
    nodes = client().sites[site].clusters[cluster].nodes.list()
    found = []
    for node in nodes:
        processor = getattr(node, "processor", None) or {}
        if not matches(processor.get("microarchitecture"), wanted):
            continue
        architecture = getattr(node, "architecture", None) or {}
        memory = getattr(node, "main_memory", None) or {}
        job_types = getattr(node, "supported_job_types", None) or {}
        found.append(
            Machine(
                site=site,
                cluster=cluster,
                uid=node.uid,
                microarchitecture=processor["microarchitecture"],
                cpu=describe_cpu(processor, architecture),
                cores=architecture.get("nb_cores") or 0,
                threads=architecture.get("nb_threads") or 0,
                sockets=architecture.get("nb_procs") or 1,
                memory_gib=(memory.get("ram_size") or 0) / 1024**3,
                gpus=len(getattr(node, "gpu_devices", None) or {}),
                exotic=bool(getattr(node, "exotic", False)),
                queues=list(job_types.get("queues") or []),
                max_walltime=job_types.get("max_walltime"),
            )
        )
    return found


def discover(wanted: str, sites: Sequence[str], workers: int) -> list[Machine]:
    """Find every matching node across ``sites``, one request per cluster."""
    pairs: list[tuple[str, str]] = []
    with futures.ThreadPoolExecutor(workers) as pool:
        listings = pool.map(
            lambda site: (site, client().sites[site].clusters.list()), sites
        )
        for site, clusters in listings:
            pairs.extend((site, cluster.uid) for cluster in clusters)

    machines: list[Machine] = []
    with futures.ThreadPoolExecutor(workers) as pool:
        results = pool.map(
            lambda pair: cluster_machines(pair[0], pair[1], wanted), pairs
        )
        for found in results:
            machines.extend(found)
    return machines


def blocking_reason(machine: Machine, walltime: int) -> str | None:
    """Why this node can never host the request, or None if it can."""
    if machine.hard_state not in RESERVABLE_STATES:
        return machine.hard_state
    if machine.max_walltime is not None and walltime > machine.max_walltime:
        return f"max walltime {machine.max_walltime // 3600}h"
    return None


def annotate_availability(
    machines: Sequence[Machine], walltime: int, now: int, workers: int
) -> None:
    """Fill in state and soonest availability, one status request per site."""
    by_site: dict[str, list[Machine]] = {}
    for machine in machines:
        by_site.setdefault(machine.site, []).append(machine)

    def site_status(site: str) -> tuple[str, dict[str, Any]]:
        statuses = client().sites[site].status
        try:
            # `waiting=yes` also reports jobs OAR has queued but not started,
            # which is what makes a *future* reservation visible here.
            status = statuses.list(waiting="yes")
        except Exception:
            status = statuses.list()
        return site, getattr(status, "nodes", None) or {}

    with futures.ThreadPoolExecutor(workers) as pool:
        for site, nodes in pool.map(site_status, by_site):
            for machine in by_site[site]:
                status = nodes.get(machine.fqdn)
                if status is None:
                    # In the reference API but not in OAR's status: retired or
                    # not yet in production, so there is nothing to reserve.
                    machine.blocked_by = "no status"
                    continue
                reservations = status.get("reservations") or []
                machine.hard_state = status.get("hard", "unknown")
                machine.soft_state = status.get("soft", "unknown")
                machine.besteffort = sum(
                    1 for r in reservations if r.get("queue") == BESTEFFORT_QUEUE
                )
                machine.reservations = len(reservations) - machine.besteffort
                machine.blocked_by = blocking_reason(machine, walltime)
                if machine.blocked_by is None:
                    machine.available_at = soonest_start(reservations, now, walltime)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def sort_key(machine: Machine) -> tuple[Any, ...]:
    # Unreservable nodes have no availability; park them at the end.
    return (
        machine.available_at is None,
        machine.available_at or 0,
        machine.site,
        machine.cluster,
        machine.uid,
    )


def format_when(machine: Machine, now: int) -> str:
    if machine.available_at is None:
        return f"unavailable ({machine.blocked_by or machine.hard_state})"
    delay = machine.available_at - now
    if delay <= 0:
        # Worth saying: a standby node has to boot before the job starts.
        return "now (standby)" if machine.hard_state == STANDBY else "now"
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(machine.available_at))
    hours, remainder = divmod(delay, 3600)
    if hours >= 24:
        days, hours = divmod(hours, 24)
        pretty = f"{days}d{hours}h"
    elif hours:
        pretty = f"{hours}h{remainder // 60:02d}m"
    else:
        pretty = f"{remainder // 60}m"
    return f"{stamp} (in {pretty})"


def render_table(machines: Sequence[Machine], now: int, walltime: str) -> str:
    header = (
        "MACHINE",
        "SITE",
        "CLUSTER",
        "MICROARCH",
        "CORES/THREADS",
        "RAM",
        "GPU",
        "QUEUE",
        f"FREE FOR {walltime}",
    )
    rows: list[tuple[str, ...]] = [header]
    for machine in sorted(machines, key=sort_key):
        rows.append(
            (
                machine.uid,
                machine.site,
                machine.cluster,
                machine.microarchitecture,
                f"{machine.cores}/{machine.threads}",
                f"{machine.memory_gib:.0f}G",
                str(machine.gpus) if machine.gpus else "-",
                machine.queue,
                format_when(machine, now),
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(rows[0])).rstrip()
    ]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows[1:]:
        lines.append(
            "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        )
    return "\n".join(lines)


def render_summary(machines: Sequence[Machine], now: int) -> str:
    lines = []
    by_cluster: dict[tuple[str, str], list[Machine]] = {}
    for machine in machines:
        by_cluster.setdefault((machine.site, machine.cluster), []).append(machine)

    lines.append("")
    lines.append("Per cluster:")
    for (site, cluster), group in sorted(
        by_cluster.items(), key=lambda item: min(sort_key(m) for m in item[1])
    ):
        free_now = sum(
            1 for m in group if m.available_at is not None and m.available_at <= now
        )
        soonest = min(
            (m for m in group if m.available_at is not None), key=sort_key, default=None
        )
        cpu = group[0].cpu
        when = format_when(soonest, now) if soonest else "none available"
        lines.append(
            f"  {site}/{cluster}: {len(group)} node(s), {free_now} free now, "
            f"soonest {when}  [{cpu}, queue={group[0].queue}]"
        )
    return "\n".join(lines)


def to_json(machines: Sequence[Machine], now: int, walltime: str) -> str:
    payload = {
        "generated_at": now,
        "walltime": walltime,
        "machines": [
            {
                "fqdn": machine.fqdn,
                "uid": machine.uid,
                "site": machine.site,
                "cluster": machine.cluster,
                "microarchitecture": machine.microarchitecture,
                "cpu": machine.cpu,
                "cores": machine.cores,
                "threads": machine.threads,
                "sockets": machine.sockets,
                "memory_gib": round(machine.memory_gib, 1),
                "gpus": machine.gpus,
                "exotic": machine.exotic,
                "queue": machine.queue,
                "queues": machine.queues,
                "max_walltime": machine.max_walltime,
                "hard_state": machine.hard_state,
                "soft_state": machine.soft_state,
                "reservations": machine.reservations,
                "besteffort_reservations": machine.besteffort,
                "blocked_by": machine.blocked_by,
                "available_at": machine.available_at,
                "available_in_seconds": (
                    max(0, machine.available_at - now)
                    if machine.available_at is not None
                    else None
                ),
            }
            for machine in sorted(machines, key=sort_key)
        ],
    }
    return json.dumps(payload, indent=2)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List Grid'5000 machines of a given microarchitecture (Zen 4 by "
            "default) across all sites, with their soonest availability."
        )
    )
    parser.add_argument(
        "--microarch",
        default="zen 4",
        help=(
            "microarchitecture prefix to match, case-insensitively "
            "(default: %(default)r, which also matches Zen 4c)"
        ),
    )
    parser.add_argument(
        "--walltime",
        default="1:00:00",
        help="length of the reservation to find a slot for (default: %(default)s)",
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        metavar="SITE",
        help="only look at these sites (default: every site)",
    )
    parser.add_argument(
        "--exclude-sites",
        nargs="+",
        default=[],
        metavar="SITE",
        help="skip these sites (e.g. ones that are unreachable)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="parallel API requests (default: %(default)s)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit JSON instead of a table"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        walltime = parse_walltime(args.walltime)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        sites = args.sites or sorted(site.uid for site in client().sites.list())
    except Exception as error:
        # A failed call carries the whole HTML error page; the status code on
        # its own is what actually tells you what went wrong.
        code = getattr(error, "response_code", None)
        detail = f"HTTP {code}" if code else str(error).splitlines()[0][:200]
        print(f"error: cannot reach the Grid'5000 API ({detail})", file=sys.stderr)
        print(
            f"hint: put your credentials in {CONFIG_FILE} or set G5K_USER/G5K_PASSWORD",
            file=sys.stderr,
        )
        return 1
    sites = [site for site in sites if site not in set(args.exclude_sites)]

    machines = discover(args.microarch, sites, args.workers)
    if not machines:
        print(
            f"no machine with a {args.microarch!r} microarchitecture found across "
            f"{len(sites)} site(s)",
            file=sys.stderr,
        )
        return 1

    now = int(time.time())
    annotate_availability(machines, walltime, now, args.workers)

    if args.json:
        print(to_json(machines, now, args.walltime))
        return 0

    print(
        f"{len(machines)} machine(s) matching {args.microarch!r} across "
        f"{len({m.site for m in machines})} site(s), "
        f"soonest slot of {args.walltime}:"
    )
    print()
    print(render_table(machines, now, args.walltime))
    print(render_summary(machines, now))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
