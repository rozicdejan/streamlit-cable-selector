import math
from typing import Dict, List

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="LV Cable Sizing Tool v4",
    page_icon="🔌",
    layout="wide",
)

st.title("LV Cable Sizing, Voltage Drop & Conduit v4 🔌")

st.markdown(
    """
This tool helps you size low-voltage cables and check existing ones using:

- **Ampacity** with derating (temperature, grouping, extra factor)
- **Voltage drop** (R-only or full **R + jX** with cosφ)
- **Short-circuit withstand check** (adiabatic, IEC-style)
- **Automatic breaker recommendation**
- **Conduit sizing** based on cable diameter and fill factor

⚠️ **Disclaimer**  
This is an engineering helper, **not** a normative design tool.  
Always verify against current standards (DIN VDE 0298-4, IEC 60364, IEC 60949, etc.) and manufacturer data before final design.
"""
)

# -----------------------------
# Data (ampacity etc.)
# -----------------------------

# Base ampacity at 30 °C, single circuit, some Cu sections (multi-core, fixed install)
BASE_AMPACITY_FROM_PDF: Dict[float, float] = {
    0.50: 3.0,
    0.75: 6.0,
    1.00: 10.0,
    1.50: 16.0,
    2.50: 20.0,
    4.00: 25.0,
}

# Ambient temperature correction factors (approx table 12-2)
TEMP_FACTORS: Dict[str, Dict[int, float]] = {
    "PVC 70°C": {
        30: 1.00,
        40: 0.87,
        50: 0.71,
        60: 0.50,
        70: 0.0,
        80: 0.0,
    },
    "XLPE 90°C": {
        30: 1.00,
        40: 0.91,
        50: 0.82,
        60: 0.71,
        70: 0.58,
        80: 0.41,
    },
}

# Bundling factors (direct on wall / in ducts, multi-core / 3-phase circuits)
BUNDLING_FACTORS_DIRECT_WALL: Dict[int, float] = {
    1: 1.00,
    2: 0.80,
    3: 0.70,
    4: 0.65,
    6: 0.57,
    10: 0.48,
}

STANDARD_SECTIONS: List[float] = [
    0.50, 0.75, 1.00, 1.50, 2.50, 4.00,
    6.00, 10.0, 16.0, 25.0, 35.0, 50.0, 70.0,
    95.0, 120.0, 150.0, 185.0, 240.0,
]

# Standard MCB / MCCB ratings [A]
STANDARD_BREAKERS = [
    6, 10, 13, 16, 20, 25, 32, 40, 50, 63, 80, 100,
    125, 160, 200, 250, 315, 400,
]

# Standard conduit internal diameters [mm] (approx.)
STANDARD_CONDUITS_MM = [16, 20, 25, 32, 40, 50, 63]


# -----------------------------
# Helper functions
# -----------------------------

def get_temp_factor(insulation: str, ambient: float) -> float:
    """Get temperature derating factor (linear interpolation in table)."""
    if insulation not in TEMP_FACTORS:
        return 1.0

    table = TEMP_FACTORS[insulation]
    ambients = sorted(table.keys())
    if ambient <= ambients[0]:
        return table[ambients[0]]
    if ambient >= ambients[-1]:
        return table[ambients[-1]]

    lower = max(a for a in ambients if a <= ambient)
    upper = min(a for a in ambients if a >= ambient)
    if lower == upper:
        return table[lower]

    f_low = table[lower]
    f_up = table[upper]
    if f_low == 0.0 or f_up == 0.0:
        return min(f_low, f_up)

    frac = (ambient - lower) / (upper - lower)
    return f_low + frac * (f_up - f_low)


def get_bundling_factor(n_circuits: int) -> float:
    """Bundling factor for 'directly on wall / in ducts'."""
    if n_circuits <= 1:
        return 1.0
    if n_circuits in BUNDLING_FACTORS_DIRECT_WALL:
        return BUNDLING_FACTORS_DIRECT_WALL[n_circuits]
    return BUNDLING_FACTORS_DIRECT_WALL[10]


def base_ampacity_for_section(
    section_mm2: float, design_current_density: float
) -> float:
    """Base ampacity at 30 °C, single circuit."""
    key = round(section_mm2, 2)
    if key in BASE_AMPACITY_FROM_PDF:
        return BASE_AMPACITY_FROM_PDF[key]
    return design_current_density * section_mm2


def resistivity_ohm_mm2_per_m(material: str) -> float:
    """DC resistivity at 20 °C in Ω·mm²/m."""
    if material == "Aluminium":
        return 0.028  # approx
    return 0.0175    # Copper (default)


def conductor_temp_from_insulation(insulation: str) -> float:
    """Approx. operating conductor temperature for impedance calculation."""
    if "XLPE" in insulation:
        return 90.0
    if "PVC" in insulation:
        return 70.0
    return 30.0  # fallback


def impedance_per_meter(section_mm2: float, material: str, insulation: str):
    """
    Return (R, X) in Ω/m for a typical LV cable at 50 Hz.

    R: DC resistance adjusted for conductor temp
    X: Typical reactance (approx) – depends weakly on size, assume generic values.
    """
    if section_mm2 <= 0:
        return 0.0, 0.0

    rho = resistivity_ohm_mm2_per_m(material)
    # R20 per km
    R20_per_km = rho / section_mm2 * 1000.0

    # Temperature coefficient
    if material == "Copper":
        alpha = 0.00393
    else:
        alpha = 0.00403

    T = conductor_temp_from_insulation(insulation)
    R_per_km = R20_per_km * (1 + alpha * (T - 20.0))
    R_per_m = R_per_km / 1000.0

    # Very rough X per km, depends a bit on size
    if section_mm2 < 16:
        X_per_km = 0.08
    elif section_mm2 < 95:
        X_per_km = 0.07
    else:
        X_per_km = 0.06
    X_per_m = X_per_km / 1000.0

    return R_per_m, X_per_m


def voltage_drop_advanced(
    system: str,
    current_a: float,
    length_m: float,
    section_mm2: float,
    material: str,
    insulation: str,
    cosphi: float,
    include_reactance: bool,
) -> float:
    """
    Voltage drop including (or excluding) reactance:

    1-phase: ΔU = 2 * I * (R cosφ + X sinφ) * L
    3-phase: ΔU = √3 * I * (R cosφ + X sinφ) * L
    """
    if current_a <= 0 or length_m <= 0 or section_mm2 <= 0:
        return 0.0

    R, X = impedance_per_meter(section_mm2, material, insulation)
    if not include_reactance:
        X = 0.0

    cosphi = max(min(cosphi, 1.0), 0.0)
    phi = math.acos(cosphi)
    sinphi = math.sin(phi)

    k = 2.0 if system == "1-phase" else math.sqrt(3)
    return k * current_a * (R * cosphi + X * sinphi) * length_m


def current_from_power_kw(
    system: str, power_kw: float, voltage_v: float, cosphi: float
) -> float:
    if power_kw <= 0 or voltage_v <= 0 or cosphi <= 0:
        return 0.0
    if system == "1-phase":
        return (power_kw * 1000) / (voltage_v * cosphi)
    else:
        return (power_kw * 1000) / (math.sqrt(3) * voltage_v * cosphi)


def power_from_current_kw(
    system: str, current_a: float, voltage_v: float, cosphi: float
) -> float:
    if system == "1-phase":
        return current_a * voltage_v * cosphi / 1000.0
    else:
        return math.sqrt(3) * current_a * voltage_v * cosphi / 1000.0


def recommend_cable_designation(
    system: str, cores: int, section_mm2: float, flex: bool
) -> str:
    """Simple EU-style cable type suggestion."""
    s_txt = f"{section_mm2:g}"
    if flex:
        return f"H07RN-F {cores}G{s_txt}"
    return f"NYM-J {cores}G{s_txt}"


# ---- Short-circuit helper ----

def k_factor_short_circuit(material: str, insulation: str) -> float:
    """
    IEC 60949 style k for adiabatic short circuit:
    Cu/PVC ~ 115, Cu/XLPE ~ 143, Al/PVC ~ 76, Al/XLPE ~ 94 (approx).
    """
    is_cu = material == "Copper"
    is_xlpe = "XLPE" in insulation or "90" in insulation
    if is_cu and is_xlpe:
        return 143.0
    if is_cu and not is_xlpe:
        return 115.0
    if not is_cu and is_xlpe:
        return 94.0
    return 76.0  # Al/PVC approx


def isc_withstand_kA(section_mm2: float, material: str, insulation: str, t_s: float) -> float:
    """Max short-circuit current [kA] the conductor can withstand for duration t_s [s]."""
    if section_mm2 <= 0 or t_s <= 0:
        return 0.0
    k = k_factor_short_circuit(material, insulation)
    return k * section_mm2 / math.sqrt(t_s) / 1000.0


# ---- Breaker helper ----

def recommend_breaker_rating(
    design_load_a: float,
    allowed_I_ampacity: float,
    breaker_util_factor: float = 1.0,
) -> int | None:
    """
    Pick smallest breaker rating that:
      - >= design_load_a
      - <= allowed_I_ampacity * breaker_util_factor
    """
    candidates = [
        In for In in STANDARD_BREAKERS
        if In >= design_load_a and In <= allowed_I_ampacity * breaker_util_factor
    ]
    if not candidates:
        return None
    return min(candidates)


# ---- Conduit helper ----

def circle_area(d_mm: float) -> float:
    """Area of circle [mm²] from diameter [mm]."""
    r = d_mm / 2.0
    return math.pi * r * r


def conduit_sizing(cables: List[dict], fill_percent: float):
    """
    cables: list of dicts with { 'count': int, 'diameter_mm': float }
    fill_percent: allowable fill (e.g. 40.0)

    Returns: (summary_df, recommended_d, recommended_fill_percent)
    """
    if fill_percent <= 0:
        fill_percent = 1.0

    # total cable area
    rows = []
    total_area = 0.0
    for i, c in enumerate(cables, start=1):
        n = max(int(c.get("count", 0)), 0)
        d = max(float(c.get("diameter_mm", 0.0)), 0.0)
        area_one = circle_area(d)
        area_total = n * area_one
        total_area += area_total
        rows.append(
            {
                "Cable type": f"type {i}",
                "Count": n,
                "Diameter [mm]": d,
                "Area one [mm²]": round(area_one, 1),
                "Total area [mm²]": round(area_total, 1),
            }
        )

    df_cables = pd.DataFrame(rows)

    if total_area <= 0:
        return df_cables, None, None, None

    required_internal_area = total_area / (fill_percent / 100.0)

    conduit_rows = []
    recommended_d = None
    recommended_fill_pct = None

    for d in STANDARD_CONDUITS_MM:
        a_conduit = circle_area(d)
        fill_ratio = total_area / a_conduit * 100.0 if a_conduit > 0 else 0.0
        ok = fill_ratio <= fill_percent

        if ok and recommended_d is None:
            recommended_d = d
            recommended_fill_pct = fill_ratio

        conduit_rows.append(
            {
                "Conduit ID [mm]": d,
                "Conduit area [mm²]": round(a_conduit, 1),
                "Fill [%]": round(fill_ratio, 1),
                "OK (≤ fill limit)": ok,
            }
        )

    df_conduit = pd.DataFrame(conduit_rows)

    # If nothing fits, pick the largest and show overfill
    if recommended_d is None and len(conduit_rows) > 0:
        largest = conduit_rows[-1]
        recommended_d = largest["Conduit ID [mm]"]
        recommended_fill_pct = largest["Fill [%]"]

    return df_cables, df_conduit, recommended_d, recommended_fill_pct


# -----------------------------
# Sidebar – global settings
# -----------------------------

with st.sidebar:
    st.header("Global inputs")

    system = st.radio(
        "System type",
        options=["3-phase", "1-phase"],
        index=0,
    )

    default_voltage = 400.0 if system == "3-phase" else 230.0
    voltage_v = st.number_input(
        "Line voltage [V]",
        min_value=50.0,
        max_value=1000.0,
        value=default_voltage,
        step=10.0,
    )

    cosphi = st.slider(
        "cos φ (power factor)",
        min_value=0.5,
        max_value=1.0,
        value=0.9,
        step=0.01,
    )

    include_reactance = st.checkbox(
        "Use impedance-based ΔV (R + jX)?",
        value=True,
        help="If disabled, use simplified R-only model.",
    )

    material = st.radio("Conductor material", ["Copper", "Aluminium"], index=0)

    insulation = st.selectbox(
        "Insulation / max conductor temp.",
        ["PVC 70°C", "XLPE 90°C", "Other / manual factor"],
        index=0,
    )

    ambient_temp = st.number_input(
        "Ambient temperature [°C]",
        min_value=-20.0,
        max_value=80.0,
        value=30.0,
        step=1.0,
    )

    if insulation == "Other / manual factor":
        temp_factor = st.number_input(
            "Temperature factor (manual)",
            min_value=0.1,
            max_value=1.0,
            value=1.0,
            step=0.01,
        )
    else:
        temp_factor = get_temp_factor(insulation, ambient_temp)
        st.info(f"Temperature factor from table: **{temp_factor:.2f}**")

    n_circuits = st.number_input(
        "Number of loaded circuits in bundle",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
    )
    bundling_factor = get_bundling_factor(int(n_circuits))
    st.info(f"Bundling factor: **{bundling_factor:.2f}**")

    additional_factor = st.number_input(
        "Additional derating factor (thermal insulation, etc.)",
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        step=0.01,
    )

    overall_derating = temp_factor * bundling_factor * additional_factor
    st.markdown(f"### Total derating factor: **{overall_derating:.3f}**")

    design_current_density = st.number_input(
        "Design current density for S > 4 mm² [A/mm²]",
        min_value=2.0,
        max_value=10.0,
        value=6.0,
        step=0.5,
    )

    st.caption("Used only where no PDF ampacity is available (S > 4 mm²).")

    breaker_curve = st.selectbox(
        "Breaker curve (info only)",
        ["B", "C", "D"],
        index=1,
        help="Used only for information in the report.",
    )

# -----------------------------
# Main modes
# -----------------------------

mode = st.radio(
    "Calculation mode",
    options=["Size cable (given load & length)", "Check existing cable"],
    index=0,
)

col_left, col_right = st.columns(2)

# -----------------------------
# MODE 1 – Size cable
# -----------------------------
if mode == "Size cable (given load & length)":
    with col_left:
        st.subheader("Input – Load & geometry")

        load_input_type = st.radio(
            "Load is given as…",
            options=["Current [A]", "Power [kW]"],
            index=0,
        )

        if load_input_type == "Current [A]":
            load_current = st.number_input(
                "Design load current [A]",
                min_value=0.1,
                max_value=2000.0,
                value=32.0,
                step=0.5,
            )
        else:
            load_power_kw = st.number_input(
                "Total active power [kW]",
                min_value=0.1,
                max_value=2000.0,
                value=15.0,
                step=0.1,
            )
            load_current = current_from_power_kw(system, load_power_kw, voltage_v, cosphi)
            st.info(f"Calculated load current: **{load_current:.1f} A**")

        length_m = st.number_input(
            "One-way cable length [m]",
            min_value=1.0,
            max_value=2000.0,
            value=50.0,
            step=1.0,
        )

        max_vdrop_percent = st.slider(
            "Max allowed voltage drop [%]",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
        )

        design_margin = st.slider(
            "Ampacity margin (I_design = I_load × (1 + margin))",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
        )

        flex = st.checkbox("Flexible cable (H07RN-F style)?", value=False)

        if system == "3-phase":
            n_cores = 5 if material == "Copper" else 4
        else:
            n_cores = 3

        # Short-circuit design inputs
        st.markdown("### Short-circuit design (for selected cable)")
        isc_pros_kA = st.number_input(
            "Prospective short-circuit current at cable start [kA]",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
        )
        sc_time_s = st.number_input(
            "Protection clearing time [s]",
            min_value=0.01,
            max_value=5.0,
            value=0.2,
            step=0.01,
        )

    # ---- Calculations ----
    required_current_for_ampacity = load_current * (1.0 + design_margin)
    required_current_after_derating = required_current_for_ampacity / max(
        overall_derating, 1e-6
    )

    rows: List[Dict[str, float]] = []
    best_section = None

    for S in STANDARD_SECTIONS:
        base_I = base_ampacity_for_section(S, design_current_density)
        allowed_I = base_I * overall_derating

        vdrop_v = voltage_drop_advanced(
            system,
            load_current,
            length_m,
            S,
            material,
            insulation,
            cosphi,
            include_reactance,
        )
        vdrop_pct = 100.0 * vdrop_v / voltage_v if voltage_v > 0 else 0.0

        ok_ampacity = allowed_I >= required_current_for_ampacity
        ok_vdrop = vdrop_pct <= max_vdrop_percent

        if ok_ampacity and ok_vdrop and best_section is None:
            best_section = S

        rows.append(
            {
                "S [mm²]": S,
                "Base I_30°C single [A]": round(base_I, 1),
                "I_allowed derated [A]": round(allowed_I, 1),
                "V_drop [V]": round(vdrop_v, 2),
                "V_drop [%]": round(vdrop_pct, 2),
                "OK ampacity": ok_ampacity,
                "OK v_drop": ok_vdrop,
            }
        )

    df = pd.DataFrame(rows)

    with col_right:
        st.subheader("Result – Recommended section")

        if best_section is None:
            st.error(
                "No standard section up to 240 mm² satisfies both **ampacity** and "
                f"**voltage drop** for {load_current:.1f} A, {length_m:.0f} m and "
                f"{max_vdrop_percent:.1f} %."
            )
        else:
            base_I_best = base_ampacity_for_section(best_section, design_current_density)
            allowed_I_best = base_I_best * overall_derating
            vdrop_best_v = voltage_drop_advanced(
                system,
                load_current,
                length_m,
                best_section,
                material,
                insulation,
                cosphi,
                include_reactance,
            )
            vdrop_best_pct = 100.0 * vdrop_best_v / voltage_v if voltage_v > 0 else 0.0

            designation = recommend_cable_designation(system, n_cores, best_section, flex)

            st.success(
                f"**Recommended minimum cross-section: {best_section:g} mm²** "
                f"({designation})"
            )
            st.markdown(
                f"""
- **Allowed current (derated)**: `{allowed_I_best:.1f} A`  
- **Required design current**: `{required_current_for_ampacity:.1f} A`  
- **Voltage drop**: `{vdrop_best_v:.2f} V` = `{vdrop_best_pct:.2f} %`
"""
            )

            # ---- Auto breaker recommendation ----
            breaker = recommend_breaker_rating(load_current, allowed_I_best, breaker_util_factor=1.0)
            st.markdown("### Breaker recommendation")

            if breaker is None:
                st.warning(
                    "No standard breaker rating both ≥ load current and ≤ cable ampacity. "
                    "Check derating or choose larger cable."
                )
            else:
                st.info(
                    f"Suggested breaker: **{breaker} A, curve {breaker_curve}** "
                    f"(In ≥ I_load, In ≤ I_cable_allowed)."
                )

            # ---- Short-circuit withstand check ----
            st.markdown("### Short-circuit withstand check (adiabatic)")

            isc_max_kA = isc_withstand_kA(best_section, material, insulation, sc_time_s)
            margin = isc_max_kA - isc_pros_kA

            st.markdown(
                f"""
For **S = {best_section:g} mm²**, material **{material}**, insulation **{insulation}**,  
and fault duration **{sc_time_s:.2f} s**:

- Max withstand short-circuit current: **{isc_max_kA:.2f} kA**
- Prospective short-circuit current: **{isc_pros_kA:.2f} kA**
"""
            )

            if margin >= 0:
                st.success(
                    f"Short-circuit check **OK** – margin ≈ `{margin:.2f} kA` "
                    "(adiabatic, no screen/PE check)."
                )
            else:
                st.error(
                    f"Short-circuit check **NOT OK** – cable rating lower by "
                    f"`{-margin:.2f} kA`. Increase section or improve protection."
                )

        st.markdown("### All sections")
        st.dataframe(df)

# -----------------------------
# MODE 2 – Check existing cable
# -----------------------------
else:
    with col_left:
        st.subheader("Input – Existing cable")

        existing_section = st.selectbox(
            "Conductor cross-section [mm²]",
            STANDARD_SECTIONS,
            index=STANDARD_SECTIONS.index(2.50) if 2.50 in STANDARD_SECTIONS else 0,
        )

        length_m = st.number_input(
            "One-way cable length [m]",
            min_value=1.0,
            max_value=2000.0,
            value=50.0,
            step=1.0,
        )

        max_vdrop_3 = st.slider(
            "Voltage drop limit #1 [%] (e.g. 3%)",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
        )
        max_vdrop_5 = st.slider(
            "Voltage drop limit #2 [%] (e.g. 5%)",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
        )

        flex = st.checkbox("Flexible cable (H07RN-F style)?", value=False)

        if system == "3-phase":
            n_cores = 5 if material == "Copper" else 4
        else:
            n_cores = 3

        # Short-circuit design inputs
        st.markdown("### Short-circuit design")
        isc_pros_kA = st.number_input(
            "Prospective short-circuit current at cable start [kA]",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
        )
        sc_time_s = st.number_input(
            "Protection clearing time [s]",
            min_value=0.01,
            max_value=5.0,
            value=0.2,
            step=0.01,
        )

    # ---- Calculations ----
    base_I = base_ampacity_for_section(existing_section, design_current_density)
    allowed_I_ampacity = base_I * overall_derating

    # For voltage-drop-limited currents, we invert ΔV formula approximately
    R_per_m, X_per_m = impedance_per_meter(existing_section, material, insulation)
    if not include_reactance:
        X_per_m = 0.0

    k_factor = 2.0 if system == "1-phase" else math.sqrt(3)
    phi = math.acos(max(min(cosphi, 1.0), 0.0))
    sinphi = math.sin(phi)

    def imax_for_vdrop(limit_percent: float) -> float:
        if limit_percent <= 0 or voltage_v <= 0 or length_m <= 0:
            return 0.0
        dv_max = (limit_percent / 100.0) * voltage_v
        denom = k_factor * (R_per_m * cosphi + X_per_m * sinphi) * length_m
        if denom <= 0:
            return 0.0
        return dv_max / denom

    I_max_vdrop_3 = imax_for_vdrop(max_vdrop_3)
    I_max_vdrop_5 = imax_for_vdrop(max_vdrop_5)

    I_design_3 = min(allowed_I_ampacity, I_max_vdrop_3)
    I_design_5 = min(allowed_I_ampacity, I_max_vdrop_5)

    P_design_3_kw = power_from_current_kw(system, I_design_3, voltage_v, cosphi)
    P_design_5_kw = power_from_current_kw(system, I_design_5, voltage_v, cosphi)

    designation = recommend_cable_designation(system, n_cores, existing_section, flex)

    with col_right:
        st.subheader("Result – Existing cable capability")

        st.success(
            f"**Cable:** {designation}  —  **S = {existing_section:g} mm²**, "
            f"{length_m:.0f} m one-way"
        )

        st.markdown(
            f"""
**Ampacity (thermal)**  

- Base ampacity 30 °C, single circuit: `{base_I:.1f} A`  
- Derated by temp/bundling/etc: **`{allowed_I_ampacity:.1f} A`**

**Voltage drop-limited currents**  

- Max current for {max_vdrop_3:.1f} % drop: `{I_max_vdrop_3:.1f} A`  
- Max current for {max_vdrop_5:.1f} % drop: `{I_max_vdrop_5:.1f} A`  

**Recommended design limits (min of ampacity & ΔV)**  

- Design current for {max_vdrop_3:.1f} %: **`{I_design_3:.1f} A`** → **`{P_design_3_kw:.1f} kW`**  
- Design current for {max_vdrop_5:.1f} %: **`{I_design_5:.1f} A`** → **`{P_design_5_kw:.1f} kW`**
"""
        )

        # ---- Breaker recommendation (based on stricter 3% limit) ----
        st.markdown("### Breaker recommendation")

        breaker = recommend_breaker_rating(I_design_3, allowed_I_ampacity, breaker_util_factor=1.0)
        if breaker is None:
            st.warning(
                "No standard breaker rating both ≥ design current and ≤ cable ampacity. "
                "Check derating or choose larger cable."
            )
        else:
            st.info(
                f"Suggested breaker for this cable: **{breaker} A, curve {breaker_curve}** "
                f"(based on {max_vdrop_3:.1f}% design limit)."
            )

        # ---- Short-circuit withstand check ----
        st.markdown("### Short-circuit withstand check (adiabatic)")

        isc_max_kA = isc_withstand_kA(existing_section, material, insulation, sc_time_s)
        margin = isc_max_kA - isc_pros_kA

        st.markdown(
            f"""
For **S = {existing_section:g} mm²**, material **{material}**, insulation **{insulation}**,  
and fault duration **{sc_time_s:.2f} s**:

- Max withstand short-circuit current: **{isc_max_kA:.2f} kA**  
- Prospective short-circuit current: **{isc_pros_kA:.2f} kA**
"""
        )

        if margin >= 0:
            st.success(
                f"Short-circuit check **OK** – margin ≈ `{margin:.2f} kA` "
                "(adiabatic, phase-core only)."
            )
        else:
            st.error(
                f"Short-circuit check **NOT OK** – cable rating lower by "
                f"`{-margin:.2f} kA`. Increase section or improve protection."
            )

        # ---- Quick "check load" tool ----
        st.markdown("---")
        st.subheader("Check a specific load on this cable")

        check_load_a = st.number_input(
            "Check load current [A]",
            min_value=0.1,
            max_value=2000.0,
            value=32.0,
            step=0.5,
        )

        vdrop_v = voltage_drop_advanced(
            system,
            check_load_a,
            length_m,
            existing_section,
            material,
            insulation,
            cosphi,
            include_reactance,
        )
        vdrop_pct = 100.0 * vdrop_v / voltage_v if voltage_v > 0 else 0.0

        ok_I = check_load_a <= allowed_I_ampacity
        ok_v3 = vdrop_pct <= max_vdrop_3
        ok_v5 = vdrop_pct <= max_vdrop_5

        st.markdown(
            f"""
- Voltage drop at {check_load_a:.1f} A: `{vdrop_v:.2f} V` = `{vdrop_pct:.2f} %`  
- **Ampacity OK?** {'✅' if ok_I else '❌'} (limit {allowed_I_ampacity:.1f} A)  
- **ΔV ≤ {max_vdrop_3:.1f} %?** {'✅' if ok_v3 else '❌'}  
- **ΔV ≤ {max_vdrop_5:.1f} %?** {'✅' if ok_v5 else '❌'}
"""
        )

# -----------------------------
# Conduit sizing helper (new)
# -----------------------------

st.markdown("---")
st.header("Conduit sizing helper 🧮")

st.write(
    "Calculate conduit diameter based on cable outer diameters and allowed fill factor."
)

conduit_col_left, conduit_col_right = st.columns(2)

with conduit_col_left:
    fill_percent = st.slider(
        "Allowed conduit fill [%]",
        min_value=20.0,
        max_value=60.0,
        value=40.0,
        step=1.0,
        help="Typical design values: 30–40 % for easy pulling. 40 % is common.",
    )

    num_types = st.number_input(
        "Number of different cable types",
        min_value=1,
        max_value=6,
        value=2,
        step=1,
    )

    cable_inputs = []
    st.subheader("Cable data")

    for i in range(num_types):
        st.markdown(f"**Cable type {i+1}**")
        count = st.number_input(
            f"Count (type {i+1})",
            min_value=0,
            max_value=100,
            value=3 if i == 0 else 0,
            step=1,
            key=f"count_{i}",
        )
        diameter = st.number_input(
            f"Outer diameter (type {i+1}) [mm]",
            min_value=0.0,
            max_value=100.0,
            value=10.0 if i == 0 else 0.0,
            step=0.1,
            key=f"diameter_{i}",
        )
        cable_inputs.append({"count": count, "diameter_mm": diameter})

with conduit_col_right:
    df_cables, df_conduit, recommended_d, recommended_fill_pct = conduit_sizing(
        cable_inputs, fill_percent
    )

    st.subheader("Cable areas")
    if df_cables is not None and len(df_cables) > 0:
        st.dataframe(df_cables)
    else:
        st.info("Enter cable counts and diameters to see results.")

    st.subheader("Conduit options")
    if df_conduit is not None and len(df_conduit) > 0:
        st.dataframe(df_conduit)
    else:
        st.info("No conduit calculation yet. Check your input.")

    if recommended_d is not None and recommended_fill_pct is not None:
        if recommended_fill_pct <= fill_percent:
            st.success(
                f"**Recommended conduit ID: {recommended_d:.0f} mm** "
                f"(fill ≈ {recommended_fill_pct:.1f} %, limit {fill_percent:.1f} %)"
            )
        else:
            st.warning(
                f"Even the largest standard conduit ({recommended_d:.0f} mm) "
                f"is over the fill limit: ≈ {recommended_fill_pct:.1f} % "
                f"(limit {fill_percent:.1f} %). Consider multiple conduits or tray."
            )
    else:
        st.info("No valid conduit recommendation – please adjust cable data.")
