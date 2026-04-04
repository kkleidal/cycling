"""
Read Garmin TCX files into pandas DataFrames, and write imputed power back.
"""

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import pandas as pd
import numpy as np

_NS_TCX = "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"
_NS_EXT = "http://www.garmin.com/xmlschemas/ActivityExtension/v2"

# Register namespaces so ET preserves the original prefixes on write.
# ET.register_namespace forbids ns\d+ prefixes, so patch the internal map directly.
ET._namespace_map.update({
    _NS_TCX:                                                          "",
    "http://www.garmin.com/xmlschemas/UserProfile/v2":               "ns2",
    _NS_EXT:                                                          "ns3",
    "http://www.garmin.com/xmlschemas/ProfileExtension/v1":          "ns4",
    "http://www.garmin.com/xmlschemas/ActivityGoals/v1":             "ns5",
    "http://www.w3.org/2001/XMLSchema-instance":                     "xsi",
})

_T = lambda tag: f"{{{_NS_TCX}}}{tag}"
_E = lambda tag: f"{{{_NS_EXT}}}{tag}"


def _parse_time(s: str) -> datetime:
    s = s.strip().rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse TCX time: {s!r}")


def parse_tcx_to_dataframe(tcx_path: str) -> pd.DataFrame:
    """
    Parse a TCX file into a DataFrame.

    Returns columns (where present):
        time, lat, lon, elevation_m, distance_km,
        speed_ms, power_w, heart_rate_bpm, cadence_rpm
    """
    tree = ET.parse(tcx_path)
    root = tree.getroot()

    rows = []
    for tp in root.iter(_T("Trackpoint")):
        row: dict = {}

        t = tp.find(_T("Time"))
        if t is not None and t.text:
            row["time"] = _parse_time(t.text)

        pos = tp.find(_T("Position"))
        if pos is not None:
            lat = pos.find(_T("LatitudeDegrees"))
            lon = pos.find(_T("LongitudeDegrees"))
            if lat is not None:
                row["lat"] = float(lat.text)
            if lon is not None:
                row["lon"] = float(lon.text)

        alt = tp.find(_T("AltitudeMeters"))
        if alt is not None:
            row["elevation_m"] = float(alt.text)

        dist = tp.find(_T("DistanceMeters"))
        if dist is not None:
            row["distance_km"] = float(dist.text) / 1000.0

        hr = tp.find(_T("HeartRateBpm"))
        if hr is not None:
            v = hr.find(_T("Value"))
            if v is not None:
                row["heart_rate_bpm"] = float(v.text)

        cad = tp.find(_T("Cadence"))
        if cad is not None:
            row["cadence_rpm"] = float(cad.text)

        tpx = tp.find(f".//{_E('TPX')}")
        if tpx is not None:
            spd = tpx.find(_E("Speed"))
            if spd is not None:
                row["speed_ms"] = float(spd.text)
            watts = tpx.find(_E("Watts"))
            if watts is not None:
                row["power_w"] = float(watts.text)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
    return df


def write_tcx_with_imputed_power(
    input_tcx_path: str,
    output_tcx_path: str,
    imputed_df: pd.DataFrame,
) -> int:
    """
    Write a new TCX file identical to *input_tcx_path* except that trackpoints
    marked as model-imputed have their <ns3:Watts> set to the imputed value.
    Trackpoints with measured power are left untouched.

    Returns the number of trackpoints patched.
    """
    # Build ISO timestamp string → imputed watts for every model-sourced row
    patch_map: dict[str, int] = {}
    for _, row in imputed_df[imputed_df["power_source"] == "model"].iterrows():
        t = row["time"]
        if hasattr(t, "strftime"):
            ts = t.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        else:
            ts = str(t).replace(" ", "T") + ".000Z"
        patch_map[ts] = max(0, int(round(float(row["power_imputed_w"]))))

    tree = ET.parse(input_tcx_path)
    root = tree.getroot()
    patched = 0

    for tp in root.iter(_T("Trackpoint")):
        t_el = tp.find(_T("Time"))
        if t_el is None or t_el.text is None:
            continue
        ts = t_el.text.strip()
        if ts not in patch_map:
            continue

        watts_val = patch_map[ts]

        # Find or create <Extensions>
        ext = tp.find(_T("Extensions"))
        if ext is None:
            ext = ET.SubElement(tp, _T("Extensions"))

        # Find or create <ns3:TPX>
        tpx = ext.find(_E("TPX"))
        if tpx is None:
            tpx = ET.SubElement(ext, _E("TPX"))

        # Find or create <ns3:Watts>
        watts_el = tpx.find(_E("Watts"))
        if watts_el is None:
            watts_el = ET.SubElement(tpx, _E("Watts"))
        watts_el.text = str(watts_val)
        patched += 1

    ET.indent(tree, space="  ")
    tree.write(output_tcx_path, xml_declaration=True, encoding="UTF-8")
    return patched
