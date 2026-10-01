#!/usr/bin/env python3
"""Rotate PDF pages clockwise without rasterizing image streams.

The rotated PDF replaces the input filename. The original input is saved beside
it as *_org.pdf before replacement.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


OBJ_RE = re.compile(rb"(?m)(\d+)\s+(\d+)\s+obj\b")
PAGE_TYPE_RE = re.compile(rb"/Type\s*/Page\b")
ROTATE_RE = re.compile(rb"/Rotate\s+(-?\d+)")
TRAILER_RE = re.compile(rb"trailer\s*(<<.*?>>)\s*startxref\s*(\d+)", re.S)


class PdfError(Exception):
    pass


def find_dict_end(data: bytes, start: int) -> int:
    depth = 0
    i = start
    while i < len(data) - 1:
        pair = data[i : i + 2]
        if pair == b"<<":
            depth += 1
            i += 2
            continue
        if pair == b">>":
            depth -= 1
            i += 2
            if depth == 0:
                return i
            continue
        i += 1
    raise PdfError("could not find end of page dictionary")


def rotate_page_body(body: bytes, degrees: int) -> bytes:
    dict_start = body.find(b"<<")
    if dict_start < 0:
        raise PdfError("page object has no dictionary")

    dict_end = find_dict_end(body, dict_start)
    page_dict = body[dict_start:dict_end]

    match = ROTATE_RE.search(page_dict)
    if match:
        old_rotate = int(match.group(1))
        new_rotate = (old_rotate + degrees) % 360
        page_dict = ROTATE_RE.sub(b"/Rotate " + str(new_rotate).encode("ascii"), page_dict, count=1)
    else:
        new_rotate = degrees % 360
        page_dict = page_dict[:-2].rstrip() + b" /Rotate " + str(new_rotate).encode("ascii") + b" >>"

    return body[:dict_start] + page_dict + body[dict_end:]


def replace_name_int(pdf_dict: bytes, name: bytes, value: int) -> bytes:
    pattern = re.compile(rb"/" + re.escape(name) + rb"\s+\d+")
    replacement = b"/" + name + b" " + str(value).encode("ascii")
    if pattern.search(pdf_dict):
        return pattern.sub(replacement, pdf_dict, count=1)
    return pdf_dict[:-2].rstrip() + b" " + replacement + b" >>"


def remove_name_int(pdf_dict: bytes, name: bytes) -> bytes:
    pattern = re.compile(rb"\s*/" + re.escape(name) + rb"\s+\d+")
    return pattern.sub(b"", pdf_dict)


def build_trailer(original: bytes, max_obj_num: int) -> bytes:
    matches = list(TRAILER_RE.finditer(original))
    if not matches:
        raise PdfError("could not find PDF trailer/startxref")

    last = matches[-1]
    trailer = last.group(1)
    prev_xref = int(last.group(2))

    size = max_obj_num + 1
    size_match = re.search(rb"/Size\s+(\d+)", trailer)
    if size_match:
        size = max(size, int(size_match.group(1)))

    trailer = remove_name_int(trailer, b"Prev")
    trailer = remove_name_int(trailer, b"XRefStm")
    trailer = replace_name_int(trailer, b"Size", size)
    return trailer[:-2].rstrip() + b" /Prev " + str(prev_xref).encode("ascii") + b" >>"


def rotated_pdf_bytes(src: Path, degrees: int) -> tuple[bytes, int]:
    original = src.read_bytes()
    if not original.startswith(b"%PDF-"):
        raise PdfError("not a PDF file")
    if re.search(rb"/Encrypt\b", original):
        raise PdfError("encrypted PDFs are not supported")

    objects = list(OBJ_RE.finditer(original))
    if not objects:
        raise PdfError("no indirect objects found")

    # An object can appear several times (original plus earlier incremental
    # updates, including earlier rotations). Only the last copy is live, so
    # rotate just that one; rotating every copy duplicates xref entries and
    # makes renderers disagree about the orientation.
    latest: dict[tuple[int, int], bytes] = {}
    max_obj_num = 0

    for index, match in enumerate(objects):
        obj_num = int(match.group(1))
        gen_num = int(match.group(2))
        max_obj_num = max(max_obj_num, obj_num)
        body_start = match.end()
        next_start = objects[index + 1].start() if index + 1 < len(objects) else len(original)
        end_match = re.search(rb"\bendobj\b", original[body_start:next_start])
        if not end_match:
            continue
        body_end = body_start + end_match.start()
        body = original[body_start:body_end]

        if PAGE_TYPE_RE.search(body):
            latest[(obj_num, gen_num)] = body
        else:
            latest.pop((obj_num, gen_num), None)  # later revision is no longer a page

    updates = [(num, gen, rotate_page_body(body, degrees)) for (num, gen), body in latest.items()]

    if not updates:
        raise PdfError("no page objects found")

    trailer = build_trailer(original, max_obj_num)
    output = bytearray(original)
    if not original.endswith(b"\n"):
        output.extend(b"\n")

    offsets: list[tuple[int, int, int]] = []
    for obj_num, gen_num, body in updates:
        offsets.append((obj_num, gen_num, len(output)))
        output.extend(f"{obj_num} {gen_num} obj\n".encode("ascii"))
        output.extend(body.strip())
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(b"xref\n")
    for obj_num, gen_num, offset in sorted(offsets):
        output.extend(f"{obj_num} 1\n".encode("ascii"))
        output.extend(f"{offset:010d} {gen_num:05d} n \n".encode("ascii"))
    output.extend(b"trailer\n")
    output.extend(trailer)
    output.extend(b"\nstartxref\n")
    output.extend(str(xref_offset).encode("ascii"))
    output.extend(b"\n%%EOF\n")

    return bytes(output), len(updates)


def backup_path(src: Path) -> Path:
    return src.with_name(f"{src.stem}_org{src.suffix}")


def rotate_pdf_in_place(src: Path, degrees: int, keep_backup: bool = True) -> int:
    if not keep_backup:
        rotated, page_count = rotated_pdf_bytes(src, degrees)
        tmp = src.with_name(src.name + f".tmp.{os.getpid()}")
        try:
            tmp.write_bytes(rotated)
            tmp.replace(src)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        print(f"{src}: rotated {degrees} degrees cw ({page_count} page{'s' if page_count != 1 else ''}); no backup kept")
        return 0

    backup = backup_path(src)
    if backup.exists():
        print(f"Skipping {src}: backup already exists ({backup})", file=sys.stderr)
        return 1

    rotated, page_count = rotated_pdf_bytes(src, degrees)
    tmp = src.with_name(src.name + f".tmp.{os.getpid()}")

    try:
        tmp.write_bytes(rotated)
        src.replace(backup)
        tmp.replace(src)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        if backup.exists() and not src.exists():
            backup.replace(src)
        raise

    print(f"{src}: rotated {degrees} degrees cw ({page_count} page{'s' if page_count != 1 else ''}); original saved as {backup}")
    return 0


def parse_degrees(value: str) -> int:
    try:
        degrees = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("DEG must be an integer") from exc

    if degrees % 90 != 0:
        raise argparse.ArgumentTypeError("DEG must be a multiple of 90")
    return degrees % 360


def main() -> int:
    parser = argparse.ArgumentParser(
        usage="%(prog)s DEG FILE [FILE ...]",
        description="Rotate PDF pages clockwise without rasterizing image streams.",
    )
    parser.add_argument("degrees", metavar="DEG", type=parse_degrees, help="clockwise degrees, usually 90 or 180")
    parser.add_argument("files", metavar="FILE", nargs="+", help="PDF files; wildcards are expanded by the shell")
    parser.add_argument("--no-backup", action="store_true", help="do not keep the original as *_org.pdf")
    args = parser.parse_args()

    status = 0
    for name in args.files:
        src = Path(name)
        if not src.is_file():
            print(f"Skipping {src}: not a regular file", file=sys.stderr)
            status = 1
            continue
        if src.suffix.lower() != ".pdf":
            print(f"Skipping {src}: not a PDF filename", file=sys.stderr)
            status = 1
            continue

        try:
            status |= rotate_pdf_in_place(src, args.degrees, keep_backup=not args.no_backup)
        except PdfError as exc:
            print(f"Failed {src}: {exc}", file=sys.stderr)
            status = 1
        except OSError as exc:
            print(f"Failed {src}: {exc}", file=sys.stderr)
            status = 1

    return status


if __name__ == "__main__":
    raise SystemExit(main())
