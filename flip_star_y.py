#!/usr/bin/env python3
"""
Flip Y coordinates in a RELION-style autopick STAR file.

Cryo-EM micrographs in MRC often use a different vertical axis than the PNG /
training view MauNet uses. After picking on PNG (or a display-oriented export),
apply a vertical flip so coordinates match the MRC stack in RELION:

    y_mrc = image_height - y_in

Usage::

    python flip_star_y.py --input picks.star --height 4096 --output picks_mrc.star
"""

from __future__ import annotations

import argparse
import os
import re
import sys


def _parse_loop_header(lines: list[str], start: int) -> tuple[dict[str, int], int]:
    """Parse loop_ column headers; return {label: 1-based index}, first data line index."""
    col_map: dict[str, int] = {}
    i = start + 1
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("_rln"):
            m = re.match(r"(_rln\S+)\s+#(\d+)", line)
            if m:
                col_map[m.group(1)] = int(m.group(2))
            i += 1
            continue
        return col_map, i
    return col_map, i


def _split_row(line: str) -> list[str]:
    return line.split()


def read_autopick_star(path: str) -> tuple[list[str], list[str], dict[str, int], list[list[str]]]:
    """Read RELION autopick STAR; return (prefix_lines, suffix_lines, col_map, data_rows)."""
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n\r") for ln in f.readlines()]

    loop_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "loop_":
            loop_idx = i
            break
    if loop_idx is None:
        raise ValueError(f"No loop_ block found in {path}")

    col_map, data_start = _parse_loop_header(lines, loop_idx)
    if "_rlnCoordinateX" not in col_map or "_rlnCoordinateY" not in col_map:
        raise ValueError(
            f"STAR must contain _rlnCoordinateX and _rlnCoordinateY (found: {list(col_map)})"
        )

    prefix = lines[:data_start]
    data_rows: list[list[str]] = []
    suffix: list[str] = []
    for line in lines[data_start:]:
        s = line.strip()
        if not s:
            if not data_rows:
                continue
            suffix.append(line)
            continue
        if s.startswith("data_") or s.startswith("loop_") or s.startswith("_rln"):
            if data_rows:
                suffix.append(line)
            continue
        data_rows.append(_split_row(s))

    if not data_rows:
        raise ValueError(f"No coordinate rows found in {path}")

    return prefix, suffix, col_map, data_rows


def flip_y_rows(
    data_rows: list[list[str]],
    col_map: dict[str, int],
    image_height: float,
    *,
    zero_indexed: bool = False,
) -> list[list[str]]:
    """Flip Y in place on copied row token lists."""
    ix = col_map["_rlnCoordinateX"] - 1
    iy = col_map["_rlnCoordinateY"] - 1
    out: list[list[str]] = []
    for row in data_rows:
        if max(ix, iy) >= len(row):
            raise ValueError(f"Row has {len(row)} fields, need X/Y at #{ix + 1} and #{iy + 1}: {row}")
        r = list(row)
        y = float(r[iy])
        if zero_indexed:
            r[iy] = str(int(round(image_height - 1 - y)))
        else:
            r[iy] = str(int(round(image_height - y)))
        out.append(r)
    return out


def write_autopick_star(
    path: str,
    prefix: list[str],
    suffix: list[str],
    data_rows: list[list[str]],
) -> None:
    lines = list(prefix)
    for row in data_rows:
        lines.append(" ".join(row))
    lines.extend(suffix)
    if lines and lines[-1].strip():
        lines.append("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def flip_star_file(
    input_path: str,
    output_path: str,
    image_height: float,
    *,
    zero_indexed: bool = False,
) -> int:
    prefix, suffix, col_map, data_rows = read_autopick_star(input_path)
    flipped = flip_y_rows(
        data_rows, col_map, image_height, zero_indexed=zero_indexed
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    write_autopick_star(output_path, prefix, suffix, flipped)
    return len(flipped)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Vertically flip Y in a RELION autopick STAR file (MRC axis fix).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--input", "-i", required=True,
        help="Input .star file (RELION autopick format)",
    )
    ap.add_argument(
        "--height", "-H", type=float, required=True,
        help="Micrograph height in pixels (MRC Ny; same units as coordinates)",
    )
    ap.add_argument(
        "--output", "-o", default=None,
        help="Output .star path (default: <input_stem>_yflip.star next to input)",
    )
    ap.add_argument(
        "--zero-indexed",
        action="store_true",
        help="Use y' = height - 1 - y instead of y' = height - y",
    )
    args = ap.parse_args()

    inp = os.path.abspath(args.input)
    if not os.path.isfile(inp):
        print(f"Error: input not found: {inp}", file=sys.stderr)
        sys.exit(1)
    if args.height <= 0:
        print("Error: --height must be positive", file=sys.stderr)
        sys.exit(1)

    if args.output:
        out = os.path.abspath(args.output)
    else:
        base, ext = os.path.splitext(inp)
        out = f"{base}_yflip{ext or '.star'}"

    n = flip_star_file(inp, out, args.height, zero_indexed=args.zero_indexed)
    print(f"Flipped {n} rows (height={args.height:g})")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
