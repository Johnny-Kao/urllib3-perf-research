from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics
import time
import tracemalloc
import zlib

from urllib3.response import GzipDecoder, GzipDecoderState


class BaselineGzipDecoder:
    """Pre-PR #5304 implementation copied from upstream baseline a164d79c."""

    def __init__(self) -> None:
        self._obj = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self._state = GzipDecoderState.FIRST_MEMBER
        self._unconsumed_tail = b""

    @property
    def has_unconsumed_tail(self) -> bool:
        return bool(self._unconsumed_tail)

    def decompress(self, data: bytes, max_length: int = -1) -> bytes:
        ret = bytearray()
        if self._state == GzipDecoderState.SWALLOW_DATA:
            return bytes(ret)
        if max_length == 0:
            self._unconsumed_tail += data
            return b""
        data = self._unconsumed_tail + data
        if not data and self._obj.eof:
            return bytes(ret)

        while True:
            try:
                ret += self._obj.decompress(
                    data, max_length=max(max_length - len(ret), 0)
                )
            except zlib.error:
                previous_state = self._state
                self._state = GzipDecoderState.SWALLOW_DATA
                self._unconsumed_tail = b""
                if previous_state == GzipDecoderState.OTHER_MEMBERS:
                    return bytes(ret)
                raise

            self._unconsumed_tail = data = (
                self._obj.unconsumed_tail or self._obj.unused_data
            )
            if max_length > 0 and len(ret) >= max_length:
                break
            if not data:
                return bytes(ret)
            if self._obj.eof:
                self._state = GzipDecoderState.OTHER_MEMBERS
                self._obj = zlib.decompressobj(16 + zlib.MAX_WBITS)

        return bytes(ret)

    def flush(self) -> bytes:
        if self._state == GzipDecoderState.SWALLOW_DATA:
            return b""
        return self._obj.flush()


def run_stream(
    decoder_cls: type[BaselineGzipDecoder] | type[GzipDecoder],
    compressed: bytes,
    chunk_sizes: list[int],
    limits: list[int],
) -> tuple[bytes, tuple[str, str] | None]:
    decoder = decoder_cls()
    out = bytearray()
    pos = 0
    i = 0
    try:
        while pos < len(compressed):
            n = chunk_sizes[i % len(chunk_sizes)]
            piece = compressed[pos : pos + n]
            pos += len(piece)
            limit = limits[i % len(limits)]
            out += decoder.decompress(piece, max_length=limit)

            guard = 0
            while decoder.has_unconsumed_tail and limit > 0:
                out += decoder.decompress(b"", max_length=limit)
                guard += 1
                if guard > 200_000:
                    raise RuntimeError("tail-drain loop")
            i += 1

        guard = 0
        while decoder.has_unconsumed_tail:
            limit = limits[i % len(limits)]
            if limit == 0:
                limit = 1
            out += decoder.decompress(b"", max_length=limit)
            i += 1
            guard += 1
            if guard > 200_000:
                raise RuntimeError("final tail-drain loop")

        out += decoder.decompress(b"")
        out += decoder.flush()
        return bytes(out), None
    except Exception as exc:
        return bytes(out), (type(exc).__name__, str(exc))


def differential_stress(cases: int = 5000) -> None:
    rng = random.Random(5304)
    sizes = [0, 1, 7, 31, 256, 1024, 8192, 65536, 262144]
    chunk_choices = [1, 2, 7, 16, 64, 256, 1024, 8192, 65536]
    limit_choices = [1, 2, 7, 64, 1024, 8192, 65536, -1]

    for rep in range(cases):
        size = rng.choice(sizes)
        raw = rng.randbytes(size)
        members = rng.choice([1, 1, 1, 2, 3, 8])
        cuts = sorted(
            [0] + [rng.randrange(size + 1) for _ in range(members - 1)] + [size]
        )
        compressed = b"".join(
            gzip.compress(raw[cuts[j] : cuts[j + 1]]) for j in range(members)
        )
        if rng.random() < 0.20:
            compressed += b"garbage"

        chunks = [rng.choice(chunk_choices) for _ in range(3)]
        limits = [rng.choice(limit_choices) for _ in range(3)]

        baseline = run_stream(BaselineGzipDecoder, compressed, chunks, limits)
        candidate = run_stream(GzipDecoder, compressed, chunks, limits)
        if baseline != candidate:
            raise AssertionError(
                "differential mismatch "
                f"case={rep} size={size} members={members} "
                f"chunks={chunks} limits={limits} "
                f"baseline=({len(baseline[0])}, {baseline[1]}) "
                f"candidate=({len(candidate[0])}, {candidate[1]})"
            )

    print(f"DIFFERENTIAL_STRESS: PASS ({cases} deterministic cases)")


def median_call_us(
    decoder_cls: type[BaselineGzipDecoder] | type[GzipDecoder],
    compressed: bytes,
    expected: bytes,
    loops: int,
) -> float:
    samples: list[float] = []
    for _ in range(loops):
        decoder = decoder_cls()
        start = time.perf_counter_ns()
        output = decoder.decompress(compressed)
        stop = time.perf_counter_ns()
        if output != expected:
            raise AssertionError("full-decode output mismatch")
        samples.append((stop - start) / 1000)
    return statistics.median(samples)


def paired_full_decode(raw: bytes, loops: int = 20, rounds: int = 9) -> dict[str, float]:
    compressed = gzip.compress(raw, compresslevel=6)
    pairs: list[tuple[float, float]] = []

    for round_no in range(rounds):
        order = (
            (BaselineGzipDecoder, GzipDecoder)
            if round_no % 2 == 0
            else (GzipDecoder, BaselineGzipDecoder)
        )
        values: dict[str, float] = {}
        for decoder_cls in order:
            values[decoder_cls.__name__] = median_call_us(
                decoder_cls, compressed, raw, loops
            )
        pairs.append((values["BaselineGzipDecoder"], values["GzipDecoder"]))

    baseline = statistics.median(x for x, _ in pairs)
    candidate = statistics.median(y for _, y in pairs)
    return {
        "decoded_bytes": float(len(raw)),
        "compressed_bytes": float(len(compressed)),
        "compression_ratio": len(raw) / len(compressed),
        "baseline_us": baseline,
        "candidate_us": candidate,
        "saved_us": baseline - candidate,
        "gain_pct": (baseline - candidate) / baseline * 100,
    }


def bounded_benchmark(raw: bytes, max_length: int, loops: int = 300) -> dict[str, float]:
    compressed = gzip.compress(raw)
    results: dict[str, list[float]] = {"BaselineGzipDecoder": [], "GzipDecoder": []}

    for round_no in range(15):
        order = (
            (BaselineGzipDecoder, GzipDecoder)
            if round_no % 2 == 0
            else (GzipDecoder, BaselineGzipDecoder)
        )
        for decoder_cls in order:
            samples: list[float] = []
            for _ in range(loops):
                decoder = decoder_cls()
                start = time.perf_counter_ns()
                output = decoder.decompress(compressed, max_length=max_length)
                stop = time.perf_counter_ns()
                if len(output) > max_length:
                    raise AssertionError("max_length contract violated")
                samples.append((stop - start) / 1000)
            results[decoder_cls.__name__].append(statistics.median(samples))

    baseline = statistics.median(results["BaselineGzipDecoder"])
    candidate = statistics.median(results["GzipDecoder"])
    return {
        "max_length": float(max_length),
        "baseline_us": baseline,
        "candidate_us": candidate,
        "gain_pct": (baseline - candidate) / baseline * 100,
    }


def multi_member_benchmark(total_bytes: int = 4 * 1024 * 1024) -> dict[str, float]:
    raw = b"x" * total_bytes
    member_size = 64 * 1024
    compressed = b"".join(
        gzip.compress(raw[i : i + member_size])
        for i in range(0, len(raw), member_size)
    )
    values = {}
    for cls in (BaselineGzipDecoder, GzipDecoder):
        samples = []
        for _ in range(35):
            decoder = cls()
            start = time.perf_counter_ns()
            output = decoder.decompress(compressed)
            stop = time.perf_counter_ns()
            if output != raw:
                raise AssertionError("multi-member mismatch")
            samples.append((stop - start) / 1000)
        values[cls.__name__] = statistics.median(samples)

    baseline = values["BaselineGzipDecoder"]
    candidate = values["GzipDecoder"]
    return {
        "members": float(total_bytes // member_size),
        "baseline_us": baseline,
        "candidate_us": candidate,
        "gain_pct": (baseline - candidate) / baseline * 100,
    }


def memory_probe(raw: bytes) -> dict[str, int]:
    compressed = gzip.compress(raw)
    out: dict[str, int] = {}
    for cls in (BaselineGzipDecoder, GzipDecoder):
        tracemalloc.start()
        decoder = cls()
        result = decoder.decompress(compressed)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if result != raw:
            raise AssertionError("memory-probe output mismatch")
        out[f"{cls.__name__}_current"] = current
        out[f"{cls.__name__}_peak"] = peak
    return out


def run_performance() -> None:
    rng = random.Random(5304)
    rows = []
    for size in (1 * 1024 * 1024, 4 * 1024 * 1024, 10 * 1024 * 1024):
        loops = 50 if size == 1 * 1024 * 1024 else 15
        rows.append(
            (
                f"{size // (1024 * 1024)}MiB-high",
                paired_full_decode(
                    (b"abcd1234" * ((size + 7) // 8))[:size], loops=loops
                ),
            )
        )
        rows.append(
            (
                f"{size // (1024 * 1024)}MiB-random",
                paired_full_decode(rng.randbytes(size), loops=loops),
            )
        )
    print("FULL_DECODE_RESULTS:")
    for name, row in rows:
        print(name, json.dumps(row, sort_keys=True))


def run_fallback() -> None:
    rng = random.Random(5304)
    bounded_raw = rng.randbytes(4 * 1024 * 1024)
    print("BOUNDED_RESULTS:")
    for limit in (1, 64, 1024, 16384, 65536, 262144):
        print(json.dumps(bounded_benchmark(bounded_raw, limit), sort_keys=True))
    print("MULTI_MEMBER_RESULT:")
    print(json.dumps(multi_member_benchmark(), sort_keys=True))


def run_memory() -> None:
    rng = random.Random(5304)
    print("MEMORY_RESULT:")
    print(json.dumps(memory_probe(rng.randbytes(10 * 1024 * 1024)), sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("all", "correctness", "performance", "fallback", "memory"),
        default="all",
    )
    parser.add_argument("--cases", type=int, default=5000)
    args = parser.parse_args()

    if args.mode in ("all", "correctness"):
        differential_stress(args.cases)
    if args.mode in ("all", "performance"):
        run_performance()
    if args.mode in ("all", "fallback"):
        run_fallback()
    if args.mode in ("all", "memory"):
        run_memory()

    print("EVIDENCE_RUN: PASS")


if __name__ == "__main__":
    main()
