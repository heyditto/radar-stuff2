"""Read one NEXRAD Level II volume from the public archive."""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_VOLUME_URI = (
    "s3://unidata-nexrad-level2/2021/11/15/KLGX/KLGX20211115_120233_V06"
)
MIN_VOLUME_BYTES = 1_000_000
ARCHIVE_BUCKET = "unidata-nexrad-level2"
VOLUME_NAME_RE = re.compile(
    r"^(?P<site>K[A-Z]{3})(?P<date>\d{8})_(?P<time>\d{6})_V\d{2}$"
)


def volume_name_time(name: str) -> datetime:
    """Parse a NEXRAD archive basename into an aware UTC datetime."""
    match = VOLUME_NAME_RE.fullmatch(Path(name).name)
    if match is None:
        raise ValueError(f"not a NEXRAD Level II volume name: {name}")
    return datetime.strptime(
        match.group("date") + match.group("time"), "%Y%m%d%H%M%S"
    ).replace(tzinfo=timezone.utc)


def volume_name_site(name: str) -> str:
    """Return the four-letter radar identifier in an archive basename."""
    match = VOLUME_NAME_RE.fullmatch(Path(name).name)
    if match is None:
        raise ValueError(f"not a NEXRAD Level II volume name: {name}")
    return match.group("site")


def list_archived_volumes(site: str, day: date) -> list[str]:
    """List data-volume basenames for one radar UTC day."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    site = site.upper()
    if not re.fullmatch(r"K[A-Z]{3}", site):
        raise ValueError(f"invalid NEXRAD site: {site}")
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    client = boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED),
    )
    prefix = f"{day:%Y/%m/%d}/{site}/"
    names: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=ARCHIVE_BUCKET, Prefix=prefix):
        for item in page.get("Contents", ()):
            name = Path(item["Key"]).name
            if VOLUME_NAME_RE.fullmatch(name):
                names.append(name)
    return sorted(names, key=volume_name_time)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` into bucket and key."""
    if not uri.startswith("s3://"):
        raise ValueError(f"expected an s3:// URI, got {uri}")
    rest = uri[5:]
    bucket, separator, key = rest.partition("/")
    if separator == "" or bucket == "" or key == "":
        raise ValueError(f"expected s3://bucket/key, got {uri}")
    return bucket, key


def download_volume(uri: str = DEFAULT_VOLUME_URI, dest: Path | None = None) -> Path:
    """Download one public Level II object with an unsigned request.

    The archive bucket is public. A signed request from an environment that
    already has AWS credentials can be rejected, so this uses
    ``signature_version=UNSIGNED``, the same idea as
    ``aws s3 cp --no-sign-request``.
    """
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from botocore.exceptions import ClientError

    bucket, key = parse_s3_uri(uri)
    # Skip the instance-metadata lookup. Unsigned public reads do not need it,
    # and a missing metadata service can stall client creation for a long time.
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    if dest is None:
        dest = Path("data") / Path(key).name
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= MIN_VOLUME_BYTES:
        return dest

    partial = dest.with_name(dest.name + ".partial")
    client = boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED),
    )
    try:
        client.download_file(bucket, key, str(partial))
    except ClientError as exc:
        partial.unlink(missing_ok=True)
        raise FileNotFoundError(f"could not download {uri}") from exc
    partial.replace(dest)
    return dest


def read_volume(path: str | Path):
    """Read a local Level II archive into a Py-ART radar object."""
    import pyart

    return pyart.io.read_nexrad_archive(str(path))


def lowest_reflectivity_sweep_index(radar) -> int:
    """Return the sweep index of the lowest tilt that contains reflectivity.

    NEXRAD split cuts store reflectivity and velocity on alternating sweeps.
    Velocity-only sweeps still carry the reflectivity field, fully masked, so
    those are skipped.
    """
    if "reflectivity" not in radar.fields:
        raise ValueError("volume has no reflectivity field")

    reflectivity = radar.fields["reflectivity"]["data"]
    best_index = None
    best_elevation = None
    for index in range(radar.nsweeps):
        sweep_slice = radar.get_slice(index)
        sweep = reflectivity[sweep_slice]
        if np.ma.getmaskarray(sweep).all():
            continue
        elevation = float(np.median(radar.elevation["data"][sweep_slice]))
        if best_elevation is None or elevation < best_elevation:
            best_index = index
            best_elevation = elevation
    if best_index is None:
        raise ValueError("no sweep contains reflectivity")
    return best_index


def sweep_elevation_deg(radar, sweep_index: int) -> float:
    """Target elevation of a sweep, in degrees."""
    fixed = getattr(radar, "fixed_angle", None)
    if isinstance(fixed, dict) and "data" in fixed:
        return float(np.asarray(fixed["data"]).ravel()[sweep_index])
    sweep_slice = radar.get_slice(sweep_index)
    return float(np.median(radar.elevation["data"][sweep_slice]))


# Split cuts and SAILS rescans share a tilt. A higher tilt sits above this gap.
# NEXRAD's 0.5° and 0.9° cuts are about 0.4° apart.
SAME_TILT_TOLERANCE_DEG = 0.15


def next_reflectivity_sweep_index(radar, lowest_index: int | None = None) -> int:
    """Return the reflectivity sweep on the next tilt above ``lowest_index``.

    The lowest sweep stays the picture being filled. Split cuts and SAILS
    rescans repeat that tilt, so they are not the sweep above. When the next
    tilt has two reflectivity cuts, the one with more unmasked gates is used.
    """
    if "reflectivity" not in radar.fields:
        raise ValueError("volume has no reflectivity field")
    if lowest_index is None:
        lowest_index = lowest_reflectivity_sweep_index(radar)

    reflectivity = radar.fields["reflectivity"]["data"]
    base = sweep_elevation_deg(radar, lowest_index)
    candidates: list[tuple[float, int, int]] = []
    for index in range(radar.nsweeps):
        sweep_slice = radar.get_slice(index)
        sweep = reflectivity[sweep_slice]
        if np.ma.getmaskarray(sweep).all():
            continue
        elevation = sweep_elevation_deg(radar, index)
        if elevation < base + SAME_TILT_TOLERANCE_DEG:
            continue
        present = int((~np.ma.getmaskarray(sweep)).sum())
        candidates.append((elevation, -present, index))
    if not candidates:
        raise ValueError("volume has no higher reflectivity sweep")

    candidates.sort()
    next_elevation = candidates[0][0]
    at_tilt = [item for item in candidates if item[0] <= next_elevation + 0.05]
    at_tilt.sort()
    return at_tilt[0][2]


def distinct_reflectivity_sweep_indices(radar) -> list[int]:
    """One sweep index per distinct reflectivity tilt, lowest elevation first.

    Sweeps within ``SAME_TILT_TOLERANCE_DEG`` are one tilt. Split cuts and
    SAILS rescans of that elevation share it. The cut with more unmasked
    reflectivity gates represents the tilt. This does not invent a tilt the
    volume does not contain.
    """
    if "reflectivity" not in radar.fields:
        raise ValueError("volume has no reflectivity field")

    reflectivity = radar.fields["reflectivity"]["data"]
    rows: list[tuple[float, int, int]] = []
    for index in range(radar.nsweeps):
        sweep_slice = radar.get_slice(index)
        sweep = reflectivity[sweep_slice]
        if np.ma.getmaskarray(sweep).all():
            continue
        present = int((~np.ma.getmaskarray(sweep)).sum())
        rows.append((sweep_elevation_deg(radar, index), -present, index))
    if not rows:
        raise ValueError("no sweep contains reflectivity")

    rows.sort()
    chosen: list[int] = []
    cluster = [rows[0]]
    anchor = rows[0][0]
    for item in rows[1:]:
        if item[0] <= anchor + SAME_TILT_TOLERANCE_DEG:
            cluster.append(item)
            continue
        cluster.sort()
        chosen.append(cluster[0][2])
        cluster = [item]
        anchor = item[0]
    cluster.sort()
    chosen.append(cluster[0][2])
    return chosen


def volume_timestamp(radar) -> str:
    """Volume start time from the Py-ART time units string, when present."""
    units = str(radar.time.get("units", ""))
    prefix = "seconds since "
    if units.startswith(prefix):
        return units[len(prefix) :]
    return units or "unknown time"


def radar_name(radar) -> str:
    """Station identifier from the volume metadata, when Py-ART recorded one."""
    for key in ("instrument_name", "radar_name"):
        value = radar.metadata.get(key)
        if value:
            return str(value).strip()
    return "UNKNOWN"
