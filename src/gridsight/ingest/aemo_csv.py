"""Parser for AEMO's multi-table CSV format (NEMweb reports and the MMSDM archive).

Each file mixes several tables. Rows are tagged in the first column:

* ``C`` comment / file header / ``END OF REPORT``
* ``I`` table header: ``I,<CATEGORY>,<REPORT>,<VERSION>,<col1>,<col2>,...``
* ``D`` data row for the most recent ``I`` header

Rows must be routed by record type and the most recent header, never by line position.
Tables are returned under the key ``<CATEGORY>_<REPORT>`` (for example ``DISPATCH_PRICE``).
Zip files, including zips nested inside zips (the NEMweb ARCHIVE format), are opened recursively.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Common MMS table names and the <CATEGORY>_<REPORT> key they appear under in the files.
MMS_ALIASES: dict[str, str] = {
    "DISPATCHPRICE": "DISPATCH_PRICE",
    "DISPATCHREGIONSUM": "DISPATCH_REGIONSUM",
    "DISPATCHINTERCONNECTORRES": "DISPATCH_INTERCONNECTORRES",
    "DISPATCHCONSTRAINT": "DISPATCH_CONSTRAINT",
    "DISPATCHLOAD": "DISPATCH_UNIT_SOLUTION",
    "DISPATCH_UNIT_SCADA": "DISPATCH_UNIT_SCADA",
    "ROOFTOP_PV_ACTUAL": "ROOFTOP_ACTUAL",
    "PREDISPATCHPRICE": "PREDISPATCH_REGION_PRICES",
    "PREDISPATCHREGIONSUM": "PREDISPATCH_REGION_SOLUTION",
    "TRADINGPRICE": "TRADING_PRICE",
}


def canonical_key(name: str) -> str:
    """Map an MMS table name or a CATEGORY_REPORT key to the CATEGORY_REPORT key."""
    upper = name.upper()
    return MMS_ALIASES.get(upper, upper)


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        non_empty = df[col].notna() & (df[col].astype(str).str.strip() != "")
        # Only convert when every non-empty value parsed as a number.
        if non_empty.any() and converted[non_empty].notna().all():
            df[col] = converted
    return df


def parse_text(text: str) -> dict[str, pd.DataFrame]:
    headers: dict[str, list[str]] = {}
    rows: dict[str, list[list[str]]] = defaultdict(list)
    current: str | None = None
    for rec in csv.reader(io.StringIO(text)):
        if not rec:
            continue
        tag = rec[0].strip().upper()
        if tag == "I" and len(rec) >= 4:
            current = f"{rec[1].strip().upper()}_{rec[2].strip().upper()}"
            cols = [c.strip().upper() for c in rec[4:]]
            if current in headers and headers[current] != cols:
                # A second version of the same table in one file: keep a separate key.
                current = f"{current}__V{rec[3].strip()}"
            headers[current] = cols
        elif tag == "D" and current is not None:
            values = rec[4:]
            ncols = len(headers[current])
            values = (values + [""] * ncols)[:ncols]
            rows[current].append(values)
    out: dict[str, pd.DataFrame] = {}
    for key, cols in headers.items():
        df = pd.DataFrame(rows.get(key, []), columns=cols, dtype=object)
        df = df.replace({"": None})
        out[key] = _coerce(df)
    return out


def parse_bytes(data: bytes, name: str = "") -> dict[str, pd.DataFrame]:
    """Parse CSV bytes or a (possibly nested) zip archive."""
    collected: dict[str, list[pd.DataFrame]] = defaultdict(list)
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                inner = zf.read(info)
                for k, v in parse_bytes(inner, info.filename).items():
                    collected[k].append(v)
    elif name.lower().endswith((".csv", ".txt")) or not name:
        for k, v in parse_text(data.decode("utf-8-sig", errors="replace")).items():
            collected[k].append(v)
    return {
        k: pd.concat(v, ignore_index=True) if len(v) > 1 else v[0] for k, v in collected.items()
    }


def parse_file(path: str | Path) -> dict[str, pd.DataFrame]:
    p = Path(path)
    return parse_bytes(p.read_bytes(), p.name)
