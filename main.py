import math
from typing import Dict, List

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="LV Cable Sizing Tool",
    page_icon="🔌",
    layout="wide",
)

st.title("LV Cable Sizing & Voltage Drop (DIN VDE style) 🔌")

st.markdown(
    """
This tool helps you size low-voltage cables and check existing ones using:

- **Ampacity** (current rating) based partly on a DIN VDE 0298-4 style table (T12 excerpt).
- **Voltage drop** for **1-phase** and **3-phase** systems.
- **Derating factors** for ambient temperature and bundling.

⚠️ **Disclaimer**  
This is an engineering helper, **not** a normative design tool.  
Always verify against the latest standards (e.g. DIN VDE 0298-4, local rules) and manufacturer data before issuing drawings or ordering cables.
"""
)

# -----------------------------
# Data from the provided PDF
# -----------------------------

# Base ampacity for one specific multi-core use-case (3 loaded cores, typical fixed installation)
# Extracted from the PDF's table 12-1 (0.5–4 mm² range, 30 °C, single circuit).
BASE_AMPACITY_FROM_PDF: Dict[float, float] = {
    0.50: 3.0,
    0.75: 6.0,
    1.00: 10.0,
    1.50: 16.0,
    2.50: 20.0,
    4.00: 25.0,
}

# Ambient temperature correction factors (from table 12-2 for 70 °C and 90 °C conductors)
TEMP_FACTORS: Dict[str, Dict[int, float]] = {
    "PVC 70°C": {
        30: 1.00,
        40: 0.87,
        50: 0.71,
        60: 0.50,
        70: 0.0,   # outside table; treated as invalid -> we clamp later
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

# Bundling factors for multi-core cables / 3-phase circuits directly bundled on wall/in ducts (table 12-6, first row)
BUNDLING_FACTORS_DIRECT_WALL: Dict[int, float] = {
    1: 1.00,
    2: 0.80,
    3: 0.70,
    4: 0.65,
    6: 0.57,
    10: 0.48,
}

# Standard LV cable cross-sections [mm²]
STANDARD_SECTIONS: List[float] = [
    0.50, 0.75, 1.00, 1.50, 2.50, 4.00,
    6.00, 10.0, 16.0, 25.0, 35.0, 50.0, 70.0,
    95.0, 120.0, 150.0, 185.0, 240.0,
]

# -----------------------------
# Helper functions
# -----------------------------


def get_temp_factor(insulation: str, ambient: float) -> float:
    """Get temperature derating factor (nearest value from table)."""
    if insulation not in TEMP_FACTORS:
        return 1.0

    table = TEMP_FACTORS[insulation]
    ambients = sorted(table.keys())
    # clamp to table range
    if ambient <= ambients[0]:
        return table[ambients[0]]
    if ambient >= ambients[-1]:
        # If we are above last tabulated value and that value is 0, show a warning later.
        return table[ambients[-1]]

    # linear interpolate between two nearest points
    lower = max(a for a in ambients if a <= ambient)
    upper = min(a for a in ambients if a >= ambient)
    if lower == upper:
        return table[lower]

    f_low = table[lower]
    f_up = table[upper]
    if f_low == 0.0 or f_up == 0.0:
        # out-of-recommended range
        return min(f_low, f_up)

    frac = (ambient - lower) / (upper - lower)
    return f_low + frac * (f_up - f_low)


def get_bundling_factor(n_circuits: int) -> float:
    """Bundling factor for 'directly on wall / in ducts' config."""
    if n_circuits <= 1:
        return 1.0
    if n_circuits in BUNDLING_FACTORS_DIRECT_WALL:
        return BUNDLING_FACTORS_DIRECT_WALL[n_circuits]
    # approximate for >10 circuits: use the 10-circuit factor
    return BUNDLING_FACTORS_DIRECT_WALL[10]


def base_ampacity_for_section(
    section_mm2: float, design_current_density: float
) -> float:
    """
    Base ampacity at 30 °C, single circuit.
    Uses PDF data where available; otherwise I ≈ J * S.
    """
    key = round(section_mm2, 2)
    if key in BASE_AMPACITY_FROM_PDF:
        return BASE_AMPACITY_FROM_PDF[key]
    return design_current_density * section_mm2


def resistivity_ohm_mm2_per_m(material: str) -> float:
    """Return DC resistivity at 20 °C in Ω·mm²/m."""
    if material == "Aluminium":
        return 0.028  # approx
    return 0.0175    # Copper (default)


def voltage_drop(
    system: str,
    current_a: float,
    length_m: float,
    section_mm2: float,
    material: str,
) -> float:
    """
    Calculate voltage drop per conductor [V] for given system.

    We use a purely resistive model (no reactance):
    - Single-phase: ΔV = 2 * I * ρ * L / S
    - Three-phase: ΔV = √3 * I * ρ * L / S
    """
    rho = resistivity_ohm_mm2_per_m(material)
    if section_mm2 <= 0:
        return 0.0

    if system == "1-phase":
        return 2 * current_a * rho * length_m / section_mm2
    else:
        return math.sqrt(3) * current_a * rho * length_m / section_mm2


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
    """
    Very rough cable type suggestion for EU practice.
    """
    s_txt = f"{section_mm2:g}"
    if flex:
        if system == "3-phase":
            return f"H07RN-F {cores}G{s_txt}"
        return f"H07RN-F {cores}G{s_txt}"

    # fixed installation
    if system == "3-phase":
        return f"NYM-J {cores}G{s_txt}"
    return f"NYM-J {cores}G{s_txt}"


# -----------------------------
# Sidebar – global settings
# -----------------------------

with st.sidebar:
    st.header("Global inputs")

    system = st.radio(
        "System type",
        options=["3-phase", "1-phase"],
        index=0,
        help="3-phase is typical 400 V in EU; 1-phase is 230 V.",
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
    st.info(f"Bundling factor (direct on wall / in ducts): **{bundling_factor:.2f}**")

    additional_factor = st.number_input(
        "Additional derating factor (e.g. thermal insulation)",
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        step=0.01,
        help="Multiply temp & bundling… e.g. 0.8 for long cable tray with poor ventilation.",
    )

    overall_derating = temp_factor * bundling_factor * additional_factor
    st.markdown(f"### Total derating factor: **{overall_derating:.3f}**")

    design_current_density = st.number_input(
        "Design current density for sections > 4 mm² [A/mm²]",
        min_value=2.0,
        max_value=10.0,
        value=6.0,
        step=0.5,
        help="Used only where no PDF ampacity is available.",
    )

    st.caption(
        "Hint: 6 A/mm² is a common rough value for Cu in air; adjust to your practice."
    )

# -----------------------------
# Main layout: two modes
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
                max_value=1000.0,
                value=32.0,
                step=0.5,
            )
        else:
            load_power_kw = st.number_input(
                "Total active power [kW]",
                min_value=0.1,
                max_value=500.0,
                value=15.0,
                step=0.1,
            )
            load_current = current_from_power_kw(system, load_power_kw, voltage_v, cosphi)
            st.info(f"Calculated load current: **{load_current:.1f} A**")

        length_m = st.number_input(
            "One-way cable length [m]",
            min_value=1.0,
            max_value=1000.0,
            value=50.0,
            step=1.0,
        )

        max_vdrop_percent = st.slider(
            "Max. allowed voltage drop [%]",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
        )

        design_margin = st.slider(
            "Ampacity design margin (I_allowed = I_load × (1 + margin))",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
        )

        flex = st.checkbox("Flexible cable (H07RN-F style)?", value=False)

        if system == "3-phase":
            n_cores = 5 if material == "Copper" else 4  # just a typical guess
        else:
            n_cores = 3

    # Calculation
    required_current_for_ampacity = load_current * (1.0 + design_margin)
    required_current_after_derating = required_current_for_ampacity / max(
        overall_derating, 1e-6
    )

    rows: List[Dict[str, float]] = []
    best_section = None

    for S in STANDARD_SECTIONS:
        base_I = base_ampacity_for_section(S, design_current_density)
        allowed_I = base_I * overall_derating

        vdrop_v = voltage_drop(system, load_current, length_m, S, material)
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
                "No standard section up to 240 mm² satisfies both **ampacity** and **voltage drop** "
                f"for {load_current:.1f} A, {length_m:.0f} m and {max_vdrop_percent:.1f}%."
            )
        else:
            base_I_best = base_ampacity_for_section(best_section, design_current_density)
            allowed_I_best = base_I_best * overall_derating
            vdrop_best_v = voltage_drop(system, load_current, length_m, best_section, material)
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

        st.markdown("### All sections")
        st.dataframe(
            df.style.highlight_min(
                subset=["S [mm²]"], color="#bdf5bd", axis=0
            ).highlight_between(
                subset=["V_drop [%]"], left=0, right=max_vdrop_percent, color="#e8ffe8"
            )
        )

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
            max_value=1000.0,
            value=50.0,
            step=1.0,
        )

        max_vdrop_3 = st.slider(
            "Voltage drop limit #1 [%] (e.g. 3 % up to main board)",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
        )
        max_vdrop_5 = st.slider(
            "Voltage drop limit #2 [%] (e.g. 5 % total)",
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

    # Calculations
    base_I = base_ampacity_for_section(existing_section, design_current_density)
    allowed_I_ampacity = base_I * overall_derating

    # Max current from voltage drop constraints
    rho = resistivity_ohm_mm2_per_m(material)
    k_factor = 2.0 if system == "1-phase" else math.sqrt(3)

    def imax_for_vdrop(limit_percent: float) -> float:
        if limit_percent <= 0 or rho <= 0 or length_m <= 0 or existing_section <= 0:
            return 0.0
        dv_max = (limit_percent / 100.0) * voltage_v
        return existing_section * dv_max / (k_factor * rho * length_m)

    I_max_vdrop_3 = imax_for_vdrop(max_vdrop_3)
    I_max_vdrop_5 = imax_for_vdrop(max_vdrop_5)

    # the real limit is the lowest of all constraints
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
**Ampacity (thermal) at given conditions**

- Base ampacity 30 °C, single circuit: `{base_I:.1f} A`
- Derated by temp/bundling/etc: **`{allowed_I_ampacity:.1f} A`**

**Voltage drop limited currents**

- Max current for {max_vdrop_3:.1f} % drop: `{I_max_vdrop_3:.1f} A`
- Max current for {max_vdrop_5:.1f} % drop: `{I_max_vdrop_5:.1f} A`

**Recommended design limits (min of ampacity & ΔV)**

- Design current for {max_vdrop_3:.1f} %: **`{I_design_3:.1f} A`** → **`{P_design_3_kw:.1f} kW`**
- Design current for {max_vdrop_5:.1f} %: **`{I_design_5:.1f} A`** → **`{P_design_5_kw:.1f} kW`**
"""
        )

        # Small "what-if" checker
        st.markdown("---")
        st.subheader("Check a specific load")

        check_load_a = st.number_input(
            "Check load current [A]",
            min_value=0.1,
            max_value=1000.0,
            value=32.0,
            step=0.5,
        )

        vdrop_v = voltage_drop(system, check_load_a, length_m, existing_section, material)
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
