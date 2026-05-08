"""
kvs_resolver.py
===============
Resolve `kvs://<StreamName>` URIs to playable HLS URLs via the AWS Kinesis
Video Streams archived-media client.

Usage:
    url = resolve_kvs("kvs://CamEntrance01", region="ap-south-1")
    # url is an HLS .m3u8 that OpenCV can open.

boto3 is imported lazily so the rest of the system runs even if it is not
installed. ``resolve_kvs`` raises ``RuntimeError`` with a clear message in
that case.
"""

from __future__ import annotations

from urllib.parse import urlparse


def is_kvs_url(url: str) -> bool:
    return isinstance(url, str) and url.strip().lower().startswith("kvs://")


def parse_kvs(url: str) -> str:
    """`kvs://CamFoo` -> stream name `CamFoo`."""
    parsed = urlparse(url)
    name = parsed.netloc or parsed.path.lstrip("/")
    if not name:
        raise ValueError(f"invalid kvs URL: {url!r}")
    return name


def resolve_kvs(
    url: str,
    region: str | None = None,
    expires_in_seconds: int = 3600,
) -> str:
    """Resolve to a live HLS playlist URL for the given KVS stream."""
    try:
        import boto3  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required to resolve kvs:// URLs. "
            "Install with: pip install boto3"
        ) from exc

    stream_name = parse_kvs(url)

    kvs = boto3.client("kinesisvideo", region_name=region)
    endpoint = kvs.get_data_endpoint(
        StreamName=stream_name,
        APIName="GET_HLS_STREAMING_SESSION_URL",
    )["DataEndpoint"]

    archived = boto3.client(
        "kinesis-video-archived-media",
        endpoint_url=endpoint,
        region_name=region,
    )
    response = archived.get_hls_streaming_session_url(
        StreamName=stream_name,
        PlaybackMode="LIVE",
        Expires=int(expires_in_seconds),
        ContainerFormat="FRAGMENTED_MP4",
    )
    return response["HLSStreamingSessionURL"]


def resolve_stream_url(url: str, region: str | None = None) -> str:
    """Pass-through for non-kvs URLs; resolve kvs:// to HLS."""
    if is_kvs_url(url):
        return resolve_kvs(url, region=region)
    return url
