import io
import time
from dataclasses import dataclass
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.dates as mdates

HOURS_PER_YEAR = 8760
QH_PER_HOUR = 4
QH_PER_YEAR = HOURS_PER_YEAR * QH_PER_HOUR
QH_DT_HOURS = 0.25
DEFAULT_YEAR = 2025
PV_ZERO_TOLERANCE_MWH = 1e-6


@dataclass
class SimulationInputs:
    batt_power_mw: float
    batt_energy_mwh: float
    pv_dc_mw: float
    productible_kwh_per_kwp: float
    pv_losses_pct: float
    plant_availability_pct: float
    eta_charge: float
    eta_discharge: float

    # Effective prices used by the optimizer/economics
    pv_price: np.ndarray
    batt_sell_price: np.ndarray
    grid_buy_price: np.ndarray

    # PV available for direct sale / standard PV-to-battery charging
    solar_profile: np.ndarray

    # PV curtailed but optionally recoverable into battery only
    curtailed_pv_recoverable_mwh: np.ndarray | None = None

    nightly_bess_revenue_eur: float = 0.0
    soc_steps: int = 101
    initial_soc_mwh: float = 0.0
    final_soc_mwh: float = 0.0
    grid_export_limit_mw: float = 0.0
    cycle_cost_eur_per_mwh: float = 0.0
    charge_quantile: float = 20.0
    discharge_quantile: float = 80.0
    max_cycles_per_day: float = 1.0
    min_spread_arbitrage_eur_per_mwh: float = 0.0

    # Capture rates
    pv_capture_rate_pct: float = 100.0
    bess_capture_rate_pct: float = 100.0

    # aFRR inputs
    enable_afrr: bool = False
    afrr_charge_price_qh: np.ndarray | None = None
    afrr_discharge_price_qh: np.ndarray | None = None
    afrr_min_spread_eur_per_mwh: float = 0.0
    afrr_cycle_cost_eur_per_mwh: float = 0.0
    afrr_max_events_per_day: int = 1
    afrr_night_start_hour: int = 20
    afrr_night_end_hour: int = 8
    afrr_pv_zero_tolerance_mwh: float = PV_ZERO_TOLERANCE_MWH
    afrr_n_qh_per_side: int = 4

    # Curtailment
    enable_tso_dso_curtailment: bool = False
    tso_dso_monthly_curtailment_pct: np.ndarray | None = None
    enable_self_curtailment: bool = False
    curtailment_threshold_eur_per_mwh: float = -1.0
    pv_commercial_structure: str = "Fully merchant"  # Fully merchant / With CfD / With PPA
    cfd_price_eur_per_mwh: float = 0.0
    negative_price_rule: bool = False
    consecutive_negative_hours_limit: int = 6
    ppa_price_eur_per_mwh: float = 0.0
    charge_battery_if_curtailment: bool = False
    enable_cfd: bool = False
    cfd_price_standalone_eur_per_mwh: float = 0.0
    enable_ppa: bool = False
    ppa_price_standalone_eur_per_mwh: float = 0.0
    project_lifetime_years: int = 1
    bess_degradation_curve_pct: np.ndarray | None = None
    degraded_bess_energy_by_year_mwh: np.ndarray | None = None

def _validate_array_length(arr: np.ndarray, name: str, expected_len: int = HOURS_PER_YEAR) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if len(arr) != expected_len:
        raise ValueError(f"{name} doit contenir exactement {expected_len} valeurs. Reçu: {len(arr)}.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contient des valeurs non numériques ou infinies.")
    return arr

def build_combined_soc_with_afrr(
    result_hourly: Dict[str, np.ndarray],
    afrr_result: Dict[str, np.ndarray] | None,
    batt_energy_mwh: float,
    initial_soc_mwh: float,
    eta_charge: float,
    eta_discharge: float,
) -> Dict[str, np.ndarray]:

    wholesale_pv_to_batt_h = np.asarray(result_hourly["pv_to_batt"], dtype=float)
    wholesale_grid_charge_h = np.asarray(result_hourly["grid_charge"], dtype=float)
    wholesale_discharge_h = np.asarray(result_hourly["discharge"], dtype=float)

    # Convert hourly → quarter-hour
    wholesale_pv_to_batt_qh = np.repeat(wholesale_pv_to_batt_h / 4.0, 4)
    wholesale_grid_charge_qh = np.repeat(wholesale_grid_charge_h / 4.0, 4)
    wholesale_discharge_market_qh = np.repeat(wholesale_discharge_h / 4.0, 4)

    if afrr_result is not None:
        afrr_charge_market_qh = np.asarray(afrr_result["afrr_charge_qh_mwh"], dtype=float)
        afrr_discharge_market_qh = np.asarray(afrr_result["afrr_discharge_qh_mwh"], dtype=float)
    else:
        afrr_charge_market_qh = np.zeros(QH_PER_YEAR, dtype=float)
        afrr_discharge_market_qh = np.zeros(QH_PER_YEAR, dtype=float)

    # Convert to SOC flows
    wholesale_charge_to_soc_qh = (wholesale_pv_to_batt_qh + wholesale_grid_charge_qh) * eta_charge
    wholesale_discharge_from_soc_qh = wholesale_discharge_market_qh / max(eta_discharge, 1e-12)

    afrr_charge_to_soc_qh = afrr_charge_market_qh * eta_charge
    afrr_discharge_from_soc_qh = afrr_discharge_market_qh / max(eta_discharge, 1e-12)

    combined_charge_to_soc_qh = wholesale_charge_to_soc_qh + afrr_charge_to_soc_qh
    combined_discharge_from_soc_qh = wholesale_discharge_from_soc_qh + afrr_discharge_from_soc_qh

    # SOC simulation
    soc_qh = np.zeros(QH_PER_YEAR + 1, dtype=float)
    soc_qh[0] = float(initial_soc_mwh)

    for t in range(QH_PER_YEAR):
        soc_next = soc_qh[t] + combined_charge_to_soc_qh[t] - combined_discharge_from_soc_qh[t]
        soc_qh[t + 1] = min(max(soc_next, 0.0), batt_energy_mwh)

    soc_hourly_end = soc_qh[4::4]

    return {
        "combined_soc_qh": soc_qh,
        "combined_soc_hourly_end": soc_hourly_end,
        "combined_charge_to_soc_qh": combined_charge_to_soc_qh,
        "combined_discharge_from_soc_qh": combined_discharge_from_soc_qh,
        "wholesale_charge_to_soc_qh": wholesale_charge_to_soc_qh,
        "wholesale_discharge_from_soc_qh": wholesale_discharge_from_soc_qh,
        "afrr_charge_to_soc_qh": afrr_charge_to_soc_qh,
        "afrr_discharge_from_soc_qh": afrr_discharge_from_soc_qh,
        "afrr_charge_market_qh": afrr_charge_market_qh,
        "afrr_discharge_market_qh": afrr_discharge_market_qh,
    }

def _read_single_column_csv(uploaded_file, expected_len: int = HOURS_PER_YEAR) -> np.ndarray:
    if uploaded_file is None:
        raise ValueError("Aucun fichier CSV fourni.")

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    raw = uploaded_file.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8-sig", errors="replace")
    else:
        text = str(raw)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Le CSV est vide.")

    values = []
    bad_rows = []

    for i, line in enumerate(lines):
        cleaned = line.strip().strip('"').strip("'").replace(",", ".")
        try:
            values.append(float(cleaned))
        except ValueError:
            bad_rows.append(i)

    if len(bad_rows) == 1 and bad_rows[0] == 0 and len(values) == expected_len:
        return np.asarray(values, dtype=float)

    if bad_rows:
        raise ValueError(
            f"Le CSV contient des valeurs non numériques dans la première colonne. "
            f"Lignes problématiques: {bad_rows[:10]}"
        )

    if len(values) != expected_len:
        raise ValueError(
            f"Le CSV doit contenir exactement {expected_len} lignes numériques. "
            f"Reçu: {len(values)}."
        )

    arr = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(arr)):
        raise ValueError("Le CSV contient des valeurs non finies.")
    return arr


def _read_single_column_csv_qh(uploaded_file, expected_len: int = QH_PER_YEAR) -> np.ndarray:
    return _read_single_column_csv(uploaded_file, expected_len=expected_len)


def read_monthly_curtailment_excel(uploaded_file) -> np.ndarray:
    if uploaded_file is None:
        raise ValueError("Aucun fichier Excel de courbe de curtailment fourni.")

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    df = pd.read_excel(uploaded_file, header=None)
    values = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=float)

    if len(values) != 12:
        raise ValueError(f"La courbe de curtailment mensuelle doit contenir exactement 12 valeurs. Reçu: {len(values)}.")

    return values

def read_bess_degradation_excel(uploaded_file, project_lifetime_years: int, initial_bess_mwh: float) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if uploaded_file is None:
        degradation_pct = np.full(project_lifetime_years, 100.0, dtype=float)
    else:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        df = pd.read_excel(uploaded_file, header=None)
        degradation_pct = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=float)

        if len(degradation_pct) < project_lifetime_years:
            raise ValueError(
                f"La courbe de dégradation BESS doit contenir au moins {project_lifetime_years} valeurs. "
                f"Reçu: {len(degradation_pct)}."
            )

        degradation_pct = degradation_pct[:project_lifetime_years]

    if len(degradation_pct) == 0:
        raise ValueError("La courbe de dégradation BESS est vide.")

    if not np.isclose(degradation_pct[0], 100.0, atol=1e-6):
        raise ValueError("La valeur de dégradation BESS en année 1 doit être égale à 100%.")

    degraded_mwh = np.zeros(project_lifetime_years, dtype=float)
    degraded_mwh[0] = float(initial_bess_mwh)

    for y in range(1, project_lifetime_years):
        degraded_mwh[y] = degraded_mwh[y - 1] * degradation_pct[y] / 100.0

    degradation_df = pd.DataFrame({
        "Year": np.arange(1, project_lifetime_years + 1),
        "Degradation_pct": degradation_pct,
        "BESS_energy_mwh": degraded_mwh,
    })

    return degradation_pct, degraded_mwh, degradation_df
    

def _make_flat_curve(value: float, expected_len: int = HOURS_PER_YEAR) -> np.ndarray:
    if value is None:
        raise ValueError("La valeur moyenne annuelle n'a pas été renseignée.")
    return np.full(expected_len, float(value), dtype=float)


def build_quarter_hour_index(year: int = DEFAULT_YEAR) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01 00:00:00", periods=QH_PER_YEAR, freq="15min")


def repeat_hourly_to_qh(hourly_arr: np.ndarray) -> np.ndarray:
    hourly_arr = np.asarray(hourly_arr, dtype=float).reshape(-1)
    if len(hourly_arr) != HOURS_PER_YEAR:
        raise ValueError(f"La série horaire doit contenir {HOURS_PER_YEAR} valeurs.")
    return np.repeat(hourly_arr, QH_PER_HOUR)


def build_night_mask_qh(idx_qh: pd.DatetimeIndex, night_start_hour: int, night_end_hour: int) -> np.ndarray:
    hours = idx_qh.hour.to_numpy()

    if night_start_hour == night_end_hour:
        return np.ones(len(idx_qh), dtype=bool)
    if night_start_hour > night_end_hour:
        return (hours >= night_start_hour) | (hours < night_end_hour)
    return (hours >= night_start_hour) & (hours < night_end_hour)


def build_standard_france_solar_profile() -> np.ndarray:
    idx = pd.date_range(f"{DEFAULT_YEAR}-01-01 00:00:00", periods=HOURS_PER_YEAR, freq="h")
    doy = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy()
    seasonal = 0.18 + 0.82 * (0.5 + 0.5 * np.sin(2 * np.pi * (doy - 81) / 365.0))
    daylight_hours = 8.0 + 8.0 * (0.5 + 0.5 * np.sin(2 * np.pi * (doy - 81) / 365.0))
    sunrise = 12.0 - daylight_hours / 2.0
    sunset = 12.0 + daylight_hours / 2.0
    shape = np.zeros(HOURS_PER_YEAR, dtype=float)

    for i in range(HOURS_PER_YEAR):
        if sunrise[i] <= hour[i] <= sunset[i]:
            x = (hour[i] - sunrise[i]) / max(sunset[i] - sunrise[i], 1e-9)
            shape[i] = (np.sin(np.pi * x) ** 1.6) * seasonal[i]

    total = shape.sum()
    if total <= 0:
        raise ValueError("Impossible de générer une courbe solaire standard valide.")
    return shape / total


def build_pv_generation_mwh(
    solar_profile_relative: np.ndarray,
    pv_dc_mw: float,
    productible_kwh_per_kwp: float,
    pv_losses_pct: float,
    plant_availability_pct: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    relative = _validate_array_length(solar_profile_relative, "Le profil solaire")
    relative = np.maximum(relative, 0.0)

    if relative.sum() <= 0:
        raise ValueError("Le profil solaire doit avoir une somme strictement positive.")
    if pv_dc_mw < 0 or productible_kwh_per_kwp < 0:
        raise ValueError("La puissance PV et le productible doivent être positifs.")
    if not (0 <= pv_losses_pct <= 100):
        raise ValueError("Les pertes PV doivent être entre 0 et 100 %.")
    if not (0 <= plant_availability_pct <= 100):
        raise ValueError("La disponibilité doit être entre 0 et 100 %.")

    annual_dc_mwh = pv_dc_mw * productible_kwh_per_kwp
    net_factor = (1.0 - pv_losses_pct / 100.0) * (plant_availability_pct / 100.0)
    annual_net_mwh = annual_dc_mwh * net_factor
    relative = relative / relative.sum()
    hourly_net_mwh = annual_net_mwh * relative

    stats = {
        "annual_dc_mwh": float(annual_dc_mwh),
        "annual_net_mwh": float(annual_net_mwh),
        "annual_losses_mwh": float(max(annual_dc_mwh - annual_net_mwh, 0.0)),
    }
    return hourly_net_mwh, stats


def apply_tso_dso_curtailment(
    pv_hourly_mwh: np.ndarray,
    monthly_curtailment_pct: np.ndarray,
) -> Dict[str, np.ndarray]:
    pv_hourly_mwh = _validate_array_length(pv_hourly_mwh, "PV horaire avant TSO/DSO")
    monthly_curtailment_pct = np.asarray(monthly_curtailment_pct, dtype=float).reshape(-1)

    if len(monthly_curtailment_pct) != 12:
        raise ValueError("La courbe mensuelle de curtailment TSO/DSO doit avoir 12 valeurs.")

    idx = pd.date_range(f"{DEFAULT_YEAR}-01-01 00:00:00", periods=HOURS_PER_YEAR, freq="h")
    month_idx = idx.month.to_numpy() - 1
    pct_hourly = monthly_curtailment_pct[month_idx]

    pv_after = pv_hourly_mwh * (1.0 - pct_hourly)
    pv_after = np.maximum(pv_after, 0.0)
    curtailed = np.maximum(pv_hourly_mwh - pv_after, 0.0)
    flag = curtailed > 1e-12

    return {
        "pv_after_tso_dso_mwh": pv_after,
        "tso_dso_curtailed_mwh": curtailed,
        "tso_dso_curtailment_flag": flag.astype(int),
        "tso_dso_monthly_pct_hourly": pct_hourly,
    }


def apply_self_curtailment(
    pv_hourly_mwh: np.ndarray,
    pv_spot_price_raw: np.ndarray,
    pv_spot_price_effective: np.ndarray,
    enable_self_curtailment: bool,
    pv_commercial_structure: str,
    curtailment_threshold_eur_per_mwh: float,
    cfd_price_eur_per_mwh: float,
    negative_price_rule: bool,
    consecutive_negative_hours_limit: int,
    ppa_price_eur_per_mwh: float,
) -> Dict[str, np.ndarray]:
    pv_hourly_mwh = _validate_array_length(pv_hourly_mwh, "PV avant self curtailment")
    pv_spot_price_raw = _validate_array_length(pv_spot_price_raw, "Prix spot PV raw")
    pv_spot_price_effective = _validate_array_length(pv_spot_price_effective, "Prix spot PV effectif")

    sellable = pv_hourly_mwh.copy()
    pv_effective_price = pv_spot_price_effective.copy()
    self_curtailed = np.zeros(HOURS_PER_YEAR, dtype=float)
    self_flag = np.zeros(HOURS_PER_YEAR, dtype=int)
    structure_arr = np.full(HOURS_PER_YEAR, pv_commercial_structure, dtype=object)
    reason_arr = np.full(HOURS_PER_YEAR, "", dtype=object)

    if not enable_self_curtailment:
        return {
            "pv_after_self_curtailment_mwh": sellable,
            "self_curtailed_mwh": self_curtailed,
            "self_curtailment_flag": self_flag,
            "pv_effective_price_eur_per_mwh": pv_effective_price,
            "pv_commercial_structure_hourly": structure_arr,
            "self_curtailment_reason": reason_arr,
        }

    if pv_commercial_structure == "Fully merchant":
        mask = pv_spot_price_raw <= curtailment_threshold_eur_per_mwh
        self_curtailed[mask] = sellable[mask]
        sellable[mask] = 0.0
        self_flag[mask] = 1
        reason_arr[mask] = "Merchant threshold curtailment"
        pv_effective_price = pv_spot_price_effective

    elif pv_commercial_structure == "With CfD":
        pv_effective_price[:] = float(cfd_price_eur_per_mwh)

        if negative_price_rule:
            neg_run = 0
            for t in range(HOURS_PER_YEAR):
                if pv_spot_price_raw[t] < 0:
                    neg_run += 1
                    if neg_run > int(consecutive_negative_hours_limit):
                        self_curtailed[t] = sellable[t]
                        sellable[t] = 0.0
                        self_flag[t] = 1
                        reason_arr[t] = "CfD negative-hours curtailment"
                else:
                    neg_run = 0

    elif pv_commercial_structure == "With PPA":
        pv_effective_price[:] = float(ppa_price_eur_per_mwh)
        mask = pv_spot_price_raw <= curtailment_threshold_eur_per_mwh
        self_curtailed[mask] = sellable[mask]
        sellable[mask] = 0.0
        self_flag[mask] = 1
        reason_arr[mask] = "PPA threshold curtailment"

    else:
        raise ValueError(f"Structure commerciale PV non reconnue: {pv_commercial_structure}")

    return {
        "pv_after_self_curtailment_mwh": sellable,
        "self_curtailed_mwh": self_curtailed,
        "self_curtailment_flag": self_flag,
        "pv_effective_price_eur_per_mwh": pv_effective_price,
        "pv_commercial_structure_hourly": structure_arr,
        "self_curtailment_reason": reason_arr,
    }


def build_pure_pv_benchmark(
    pv_generation_mwh: np.ndarray,
    pv_price: np.ndarray,
    grid_export_limit_mw: float,
) -> Dict[str, np.ndarray]:
    pv_generation_mwh = _validate_array_length(pv_generation_mwh, "Production PV benchmark")
    pv_price = _validate_array_length(pv_price, "Prix PV benchmark")

    pv_only_direct_mwh = np.minimum(np.maximum(pv_generation_mwh, 0.0), float(grid_export_limit_mw))
    pv_only_revenue_eur = pv_only_direct_mwh * pv_price
    total_pv_only_revenue_eur = float(pv_only_revenue_eur.sum())

    return {
        "pv_only_direct_mwh": pv_only_direct_mwh,
        "pv_only_revenue_eur": pv_only_revenue_eur,
        "total_pv_only_revenue_eur": np.array([total_pv_only_revenue_eur]),
    }


def optimize_dispatch_dp(inputs: SimulationInputs) -> Dict[str, np.ndarray]:
    pv_sellable = _validate_array_length(inputs.solar_profile, "La production PV nette horaire sellable")
    pv_sellable = np.maximum(pv_sellable, 0.0)

    if inputs.curtailed_pv_recoverable_mwh is None:
        pv_recoverable = np.zeros(HOURS_PER_YEAR, dtype=float)
    else:
        pv_recoverable = _validate_array_length(inputs.curtailed_pv_recoverable_mwh, "PV curtailed recoverable")
        pv_recoverable = np.maximum(pv_recoverable, 0.0)

    pv_price = _validate_array_length(inputs.pv_price, "Le prix PV")
    batt_sell = _validate_array_length(inputs.batt_sell_price, "Le prix de vente batterie")
    grid_buy = _validate_array_length(inputs.grid_buy_price, "Le prix d'achat réseau")

    idx = pd.date_range(f"{DEFAULT_YEAR}-01-01 00:00:00", periods=HOURS_PER_YEAR, freq="h")

    df_thresholds = pd.DataFrame({
        "datetime": idx,
        "grid_buy": grid_buy,
        "batt_sell": batt_sell,
    })
    df_thresholds["day"] = df_thresholds["datetime"].dt.date

    charge_threshold_series = df_thresholds.groupby("day")["grid_buy"].transform(
        lambda x: np.percentile(x, inputs.charge_quantile)
    ).to_numpy()

    discharge_threshold_series = df_thresholds.groupby("day")["batt_sell"].transform(
        lambda x: np.percentile(x, inputs.discharge_quantile)
    ).to_numpy()

    if np.any(~np.isfinite(pv_sellable)) or np.any(~np.isfinite(pv_price)) or np.any(~np.isfinite(batt_sell)) or np.any(~np.isfinite(grid_buy)):
        raise ValueError("Une ou plusieurs séries contiennent des valeurs invalides.")
    if inputs.batt_power_mw < 0 or inputs.batt_energy_mwh < 0:
        raise ValueError("La puissance et la capacité batterie doivent être positives.")
    if inputs.eta_charge <= 0 or inputs.eta_charge > 1:
        raise ValueError("Le rendement de charge doit être compris entre 0 et 1.")
    if inputs.eta_discharge <= 0 or inputs.eta_discharge > 1:
        raise ValueError("Le rendement de décharge doit être compris entre 0 et 1.")
    if inputs.initial_soc_mwh < 0 or inputs.final_soc_mwh < 0:
        raise ValueError("Les SOC initial et final doivent être positifs.")
    if inputs.initial_soc_mwh > inputs.batt_energy_mwh:
        raise ValueError("Le SOC initial ne peut pas dépasser la capacité batterie.")
    if inputs.final_soc_mwh > inputs.batt_energy_mwh:
        raise ValueError("Le SOC final ne peut pas dépasser la capacité batterie.")

    T = len(pv_sellable)
    if T != HOURS_PER_YEAR:
        raise ValueError("Toutes les séries doivent contenir 8760 heures.")

    soc_steps = int(max(21, inputs.soc_steps))
    soc_grid = np.linspace(0.0, inputs.batt_energy_mwh, soc_steps)

    def nearest_state_index(value: float) -> int:
        value = min(max(value, 0.0), inputs.batt_energy_mwh)
        return int(np.argmin(np.abs(soc_grid - value)))

    init_idx = nearest_state_index(inputs.initial_soc_mwh)
    final_idx = nearest_state_index(inputs.final_soc_mwh)

    DT = 1.0
    charge_soc_max = inputs.batt_power_mw * inputs.eta_charge * DT
    discharge_soc_max = inputs.batt_power_mw * DT / inputs.eta_discharge

    transitions = []
    for i, soc in enumerate(soc_grid):
        j_min = np.searchsorted(soc_grid, max(0.0, soc - discharge_soc_max), side="left")
        j_max = np.searchsorted(soc_grid, min(inputs.batt_energy_mwh, soc + charge_soc_max), side="right") - 1
        transitions.append(np.arange(j_min, j_max + 1, dtype=int))

    def run_dp_once(required_discharge_price_estimate: np.ndarray) -> Dict[str, np.ndarray]:
        neg_inf = -1e30
        value_next = np.full(soc_steps, neg_inf, dtype=float)
        value_next[final_idx] = 0.0
        policy_next = np.full((T, soc_steps), -1, dtype=np.int16 if soc_steps < 32000 else np.int32)

        estimate_gate = np.asarray(required_discharge_price_estimate, dtype=float).reshape(-1)
        if len(estimate_gate) != T:
            raise ValueError("La courbe estimée de prix requis de décharge a une mauvaise longueur.")
        estimate_gate = np.nan_to_num(estimate_gate, nan=-1e30, posinf=1e30, neginf=-1e30)

        for t in range(T - 1, -1, -1):
            value_now = np.full(soc_steps, neg_inf, dtype=float)
            pv_sellable_t = pv_sellable[t]
            pv_recoverable_t = pv_recoverable[t]
            pv_price_t = pv_price[t]
            batt_sell_t = batt_sell[t]
            grid_buy_t = grid_buy[t]

            for i in range(soc_steps):
                best_val = neg_inf
                best_j = -1
                soc_i = soc_grid[i]

                for j in transitions[i]:
                    delta_soc = soc_grid[j] - soc_i

                    pv_direct_candidate = pv_sellable_t
                    sellable_pv_to_batt = 0.0
                    recoverable_pv_to_batt = 0.0
                    grid_charge = 0.0
                    discharge_candidate = 0.0
                    cycle_penalty = 0.0

                    if delta_soc > 1e-12:
                        charge_input = delta_soc / inputs.eta_charge

                        recoverable_pv_to_batt = min(charge_input, pv_recoverable_t)
                        remaining_after_recoverable = charge_input - recoverable_pv_to_batt

                        sellable_pv_to_batt = min(remaining_after_recoverable, pv_sellable_t)
                        remaining_after_sellable = remaining_after_recoverable - sellable_pv_to_batt

                        grid_charge = max(remaining_after_sellable, 0.0)
                        pv_direct_candidate = pv_sellable_t - sellable_pv_to_batt

                        if grid_charge > 1e-9 and grid_buy_t > charge_threshold_series[t]:
                            continue

                        if (recoverable_pv_to_batt + sellable_pv_to_batt) < charge_input and (batt_sell_t - grid_buy_t) < inputs.min_spread_arbitrage_eur_per_mwh:
                            continue

                    elif delta_soc < -1e-12:
                        discharge_candidate = (-delta_soc) * inputs.eta_discharge

                        if discharge_candidate > 1e-9:
                            if batt_sell_t < discharge_threshold_series[t]:
                                continue
                            if batt_sell_t < estimate_gate[t]:
                                continue

                    total_export = pv_direct_candidate + discharge_candidate

                    if total_export > inputs.grid_export_limit_mw:
                        excess = total_export - inputs.grid_export_limit_mw
                        reduction_pv = min(excess, pv_direct_candidate)
                        pv_direct_candidate -= reduction_pv
                        excess -= reduction_pv

                        if excess > 0:
                            discharge_candidate = max(discharge_candidate - excess, 0.0)

                        if discharge_candidate > 0:
                            throughput = abs(delta_soc)
                            cycle_penalty = (throughput / max(inputs.batt_energy_mwh, 1e-12)) * inputs.cycle_cost_eur_per_mwh

                    reward = pv_direct_candidate * pv_price_t

                    if delta_soc > 1e-12:
                        reward -= grid_charge * grid_buy_t
                    elif delta_soc < -1e-12:
                        reward += discharge_candidate * batt_sell_t
                        reward -= cycle_penalty

                    total_val = reward + value_next[j]
                    if total_val > best_val:
                        best_val = total_val
                        best_j = int(j)

                value_now[i] = best_val
                policy_next[t, i] = best_j

            value_next = value_now

        if np.all(value_next == neg_inf):
            raise RuntimeError("DP failed: all states unreachable")

        soc = np.zeros(T + 1, dtype=float)
        soc[0] = soc_grid[init_idx]
        state = init_idx

        pv_direct = np.zeros(T, dtype=float)
        pv_to_batt = np.zeros(T, dtype=float)
        pv_curtailed_to_battery = np.zeros(T, dtype=float)
        grid_charge = np.zeros(T, dtype=float)
        discharge = np.zeros(T, dtype=float)
        batt_sale_revenue = np.zeros(T, dtype=float)
        grid_charge_cost = np.zeros(T, dtype=float)
        pv_direct_revenue = np.zeros(T, dtype=float)
        avg_stored_charge_price = np.full(T + 1, np.nan, dtype=float)
        required_discharge_price = np.full(T, np.nan, dtype=float)
        stored_energy_value_eur = 0.0
        stored_energy_mwh = soc[0]

        avg_stored_charge_price[0] = 0.0 if stored_energy_mwh > 1e-9 else np.nan

        for t in range(T):
            next_state = int(policy_next[t, state])
            if next_state < 0:
                raise RuntimeError(f"Policy failure at t={t}, state={state}")

            delta_soc = soc_grid[next_state] - soc_grid[state]
            soc[t + 1] = soc_grid[next_state]

            pv_sellable_t = pv_sellable[t]
            pv_recoverable_t = pv_recoverable[t]

            pv_direct_candidate = pv_sellable_t
            sellable_pv_to_batt = 0.0
            recoverable_pv_to_batt = 0.0
            grid_charge[t] = 0.0
            discharge[t] = 0.0

            if delta_soc > 1e-12:
                charge_input = delta_soc / inputs.eta_charge

                recoverable_pv_to_batt = min(charge_input, pv_recoverable_t)
                remaining_after_recoverable = charge_input - recoverable_pv_to_batt

                sellable_pv_to_batt = min(remaining_after_recoverable, pv_sellable_t)
                remaining_after_sellable = remaining_after_recoverable - sellable_pv_to_batt

                grid_charge[t] = max(remaining_after_sellable, 0.0)
                pv_direct_candidate = pv_sellable_t - sellable_pv_to_batt

            elif delta_soc < -1e-12:
                discharge[t] = (-delta_soc) * inputs.eta_discharge

            pv_to_batt[t] = sellable_pv_to_batt
            pv_curtailed_to_battery[t] = recoverable_pv_to_batt

            if delta_soc > 1e-12:
                charge_cost_eur = (
                    sellable_pv_to_batt * pv_price[t] +
                    grid_charge[t] * grid_buy[t]
                    # recoverable_pv_to_batt enters at zero opportunity cost
                )
                stored_energy_value_eur += charge_cost_eur
                stored_energy_mwh += delta_soc

            elif delta_soc < -1e-12:
                avg_cost_now = stored_energy_value_eur / max(stored_energy_mwh, 1e-9)
                energy_removed_from_soc = -delta_soc
                cost_removed_eur = avg_cost_now * energy_removed_from_soc
                stored_energy_value_eur = max(stored_energy_value_eur - cost_removed_eur, 0.0)
                stored_energy_mwh = max(stored_energy_mwh - energy_removed_from_soc, 0.0)

            if stored_energy_mwh > 1e-9:
                avg_stored_charge_price[t + 1] = stored_energy_value_eur / stored_energy_mwh
            else:
                avg_stored_charge_price[t + 1] = np.nan

            if np.isfinite(avg_stored_charge_price[t]):
                required_discharge_price[t] = avg_stored_charge_price[t] + inputs.min_spread_arbitrage_eur_per_mwh

            total_export = pv_direct_candidate + discharge[t]
            if total_export > inputs.grid_export_limit_mw:
                excess = total_export - inputs.grid_export_limit_mw
                reduction_pv = min(excess, pv_direct_candidate)
                pv_direct_candidate -= reduction_pv
                excess -= reduction_pv
                if excess > 0:
                    discharge[t] = max(discharge[t] - excess, 0.0)

            pv_direct[t] = max(pv_direct_candidate, 0.0)
            pv_direct_revenue[t] = pv_direct[t] * pv_price[t]
            batt_sale_revenue[t] = discharge[t] * batt_sell[t]
            grid_charge_cost[t] = grid_charge[t] * grid_buy[t]
            state = next_state

        total_direct_pv_revenue = float(pv_direct_revenue.sum())
        total_batt_sale_revenue = float(batt_sale_revenue.sum())
        total_grid_charge_cost = float(grid_charge_cost.sum())
        nightly_revenue_total = float(inputs.nightly_bess_revenue_eur * (T // 24))
        total_revenue = total_direct_pv_revenue + total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total

        return {
            "soc": soc,
            "pv_direct": pv_direct,
            "pv_to_batt": pv_to_batt,
            "pv_curtailed_to_battery": pv_curtailed_to_battery,
            "grid_charge": grid_charge,
            "discharge": discharge,
            "pv_direct_revenue": pv_direct_revenue,
            "batt_sale_revenue": batt_sale_revenue,
            "grid_charge_cost": grid_charge_cost,
            "total_direct_pv_revenue": np.array([total_direct_pv_revenue]),
            "total_batt_sale_revenue": np.array([total_batt_sale_revenue]),
            "total_grid_charge_cost": np.array([total_grid_charge_cost]),
            "nightly_revenue_total": np.array([nightly_revenue_total]),
            "total_revenue": np.array([total_revenue]),
            "equivalent_cycles": np.array([discharge.sum() / max(inputs.batt_energy_mwh, 1e-12)]),
            "energy_sold_total_mwh": np.array([pv_direct.sum() + discharge.sum()]),
            "energy_shifted_mwh": np.array([discharge.sum()]),
            "pv_direct_sold_mwh": np.array([pv_direct.sum()]),
            "avg_stored_charge_price": avg_stored_charge_price,
            "required_discharge_price": required_discharge_price,
            "hourly_datetime": idx,
            "required_discharge_price_gate_estimate": estimate_gate,
        }

    n_passes = 3
    required_estimate = np.full(T, -1e30, dtype=float)
    final_result = None

    for _ in range(n_passes):
        candidate = run_dp_once(required_estimate)
        new_estimate = np.nan_to_num(candidate["required_discharge_price"], nan=-1e30, posinf=1e30, neginf=-1e30)

        if final_result is not None and np.allclose(new_estimate, required_estimate, atol=1e-6, rtol=0.0):
            final_result = candidate
            break

        final_result = candidate
        required_estimate = new_estimate.copy()

    return final_result


def _afrr_qh_limits(
    batt_power_mw: float,
    eta_charge: float,
    eta_discharge: float,
    dt_hours: float = QH_DT_HOURS,
) -> Dict[str, float]:
    input_per_qh = batt_power_mw * dt_hours
    stored_per_qh = input_per_qh * eta_charge
    output_per_qh = stored_per_qh * eta_discharge
    return {
        "input_per_qh_mwh": float(input_per_qh),
        "stored_per_qh_mwh": float(stored_per_qh),
        "output_per_qh_mwh": float(output_per_qh),
    }


def select_best_daily_afrr_trade_blocks(
    charge_prices_day: np.ndarray,
    discharge_prices_day: np.ndarray,
    eligible_mask_day: np.ndarray,
    batt_power_mw: float,
    batt_energy_mwh: float,
    eta_charge: float,
    eta_discharge: float,
    afrr_cycle_cost_eur_per_mwh: float,
    afrr_min_spread_eur_per_mwh: float,
    n_qh: int = 4,
    dt_hours: float = QH_DT_HOURS,
) -> Dict[str, object]:
    idx_eligible = np.where(eligible_mask_day)[0]

    if len(idx_eligible) < 2 * n_qh:
        return {
            "execute": False,
            "charge_indices": [],
            "discharge_indices": [],
            "avg_charge_price": np.nan,
            "avg_discharge_price": np.nan,
            "expected_net_spread_eur_per_mwh": np.nan,
            "expected_charge_input_mwh": 0.0,
            "expected_stored_mwh": 0.0,
            "expected_discharge_output_mwh": 0.0,
            "reason": "Pas assez de quarts d'heure éligibles.",
        }

    best = {
        "execute": False,
        "charge_indices": [],
        "discharge_indices": [],
        "avg_charge_price": np.nan,
        "avg_discharge_price": np.nan,
        "expected_net_spread_eur_per_mwh": -np.inf,
        "expected_charge_input_mwh": 0.0,
        "expected_stored_mwh": 0.0,
        "expected_discharge_output_mwh": 0.0,
        "reason": "Aucune combinaison valide.",
    }

    power_limited_input_per_qh = batt_power_mw * dt_hours
    total_charge_input_mwh = n_qh * power_limited_input_per_qh
    total_stored_mwh = total_charge_input_mwh * eta_charge
    total_discharge_output_mwh = total_stored_mwh * eta_discharge

    for split_pos in range(1, len(idx_eligible)):
        charge_pool = idx_eligible[:split_pos]
        discharge_pool = idx_eligible[split_pos:]

        if len(charge_pool) < n_qh or len(discharge_pool) < n_qh:
            continue

        charge_sorted = charge_pool[np.argsort(charge_prices_day[charge_pool])]
        selected_charge = np.sort(charge_sorted[:n_qh])

        discharge_sorted = discharge_pool[np.argsort(-discharge_prices_day[discharge_pool])]
        selected_discharge = np.sort(discharge_sorted[:n_qh])

        if len(selected_charge) < n_qh or len(selected_discharge) < n_qh:
            continue
        if selected_charge.max() >= selected_discharge.min():
            continue

        avg_charge_price = float(np.mean(charge_prices_day[selected_charge]))
        avg_discharge_price = float(np.mean(discharge_prices_day[selected_discharge]))

        effective_input_cost_per_mwh_out = avg_charge_price / max(eta_charge * eta_discharge, 1e-12)
        net_spread = avg_discharge_price - effective_input_cost_per_mwh_out - afrr_cycle_cost_eur_per_mwh

        if net_spread > best["expected_net_spread_eur_per_mwh"]:
            best = {
                "execute": net_spread >= afrr_min_spread_eur_per_mwh,
                "charge_indices": selected_charge.tolist(),
                "discharge_indices": selected_discharge.tolist(),
                "avg_charge_price": avg_charge_price,
                "avg_discharge_price": avg_discharge_price,
                "expected_net_spread_eur_per_mwh": float(net_spread),
                "expected_charge_input_mwh": float(total_charge_input_mwh),
                "expected_stored_mwh": float(total_stored_mwh),
                "expected_discharge_output_mwh": float(total_discharge_output_mwh),
                "reason": "OK" if net_spread >= afrr_min_spread_eur_per_mwh else "Spread insuffisant.",
            }

    if best["expected_net_spread_eur_per_mwh"] < afrr_min_spread_eur_per_mwh:
        best["execute"] = False

    return best


def simulate_afrr_night_arbitrage(inputs: SimulationInputs, result_hourly: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if not inputs.enable_afrr:
        return {
            "afrr_charge_qh_mwh": np.zeros(QH_PER_YEAR, dtype=float),
            "afrr_discharge_qh_mwh": np.zeros(QH_PER_YEAR, dtype=float),
            "afrr_soc_qh": np.repeat(result_hourly["soc"][:-1], QH_PER_HOUR).astype(float),
            "afrr_charge_cost_qh_eur": np.zeros(QH_PER_YEAR, dtype=float),
            "afrr_sale_revenue_qh_eur": np.zeros(QH_PER_YEAR, dtype=float),
            "afrr_cycle_cost_qh_eur": np.zeros(QH_PER_YEAR, dtype=float),
            "afrr_net_revenue_qh_eur": np.zeros(QH_PER_YEAR, dtype=float),
            "afrr_daily_log": pd.DataFrame(),
        }

    if inputs.afrr_charge_price_qh is None or inputs.afrr_discharge_price_qh is None:
        raise ValueError("Les courbes de prix aFRR quart-horaires doivent être fournies si aFRR est activé.")

    charge_prices_qh = _validate_array_length(inputs.afrr_charge_price_qh, "Prix aFRR charge", QH_PER_YEAR)
    discharge_prices_qh = _validate_array_length(inputs.afrr_discharge_price_qh, "Prix aFRR décharge", QH_PER_YEAR)

    idx_qh = build_quarter_hour_index(DEFAULT_YEAR)
    pv_hourly = _validate_array_length(inputs.solar_profile, "Production PV nette horaire", HOURS_PER_YEAR)
    pv_qh = repeat_hourly_to_qh(pv_hourly / 4.0)

    night_mask_qh = build_night_mask_qh(idx_qh, inputs.afrr_night_start_hour, inputs.afrr_night_end_hour)
    no_pv_mask_qh = pv_qh <= float(inputs.afrr_pv_zero_tolerance_mwh)
    eligible_mask_qh = night_mask_qh & no_pv_mask_qh

    afrr_charge_qh_mwh = np.zeros(QH_PER_YEAR, dtype=float)
    afrr_discharge_qh_mwh = np.zeros(QH_PER_YEAR, dtype=float)
    afrr_charge_cost_qh_eur = np.zeros(QH_PER_YEAR, dtype=float)
    afrr_sale_revenue_qh_eur = np.zeros(QH_PER_YEAR, dtype=float)
    afrr_cycle_cost_qh_eur = np.zeros(QH_PER_YEAR, dtype=float)
    afrr_net_revenue_qh_eur = np.zeros(QH_PER_YEAR, dtype=float)
    afrr_soc_qh = np.zeros(QH_PER_YEAR, dtype=float)

    daily_logs = []
    soc_current = float(result_hourly["soc"][0])

    df = pd.DataFrame({
        "datetime": idx_qh,
        "charge_price": charge_prices_qh,
        "discharge_price": discharge_prices_qh,
        "eligible": eligible_mask_qh,
        "pv_qh_mwh": pv_qh,
    })
    df["day"] = df["datetime"].dt.date

    limits = _afrr_qh_limits(inputs.batt_power_mw, inputs.eta_charge, inputs.eta_discharge, QH_DT_HOURS)

    for day, group in df.groupby("day", sort=True):
        group_idx = group.index.to_numpy()
        charge_day = group["charge_price"].to_numpy(dtype=float)
        discharge_day = group["discharge_price"].to_numpy(dtype=float)
        eligible_day = group["eligible"].to_numpy(dtype=bool)

        best_trade = select_best_daily_afrr_trade_blocks(
            charge_prices_day=charge_day,
            discharge_prices_day=discharge_day,
            eligible_mask_day=eligible_day,
            batt_power_mw=inputs.batt_power_mw,
            batt_energy_mwh=inputs.batt_energy_mwh,
            eta_charge=inputs.eta_charge,
            eta_discharge=inputs.eta_discharge,
            afrr_cycle_cost_eur_per_mwh=inputs.afrr_cycle_cost_eur_per_mwh,
            afrr_min_spread_eur_per_mwh=inputs.afrr_min_spread_eur_per_mwh,
            n_qh=inputs.afrr_n_qh_per_side,
            dt_hours=QH_DT_HOURS,
        )

        soc_day_start = soc_current
        day_charge_qh_mwh = np.zeros(len(group_idx), dtype=float)
        day_discharge_qh_mwh = np.zeros(len(group_idx), dtype=float)
        day_charge_cost_qh_eur = np.zeros(len(group_idx), dtype=float)
        day_sale_revenue_qh_eur = np.zeros(len(group_idx), dtype=float)
        day_cycle_cost_qh_eur = np.zeros(len(group_idx), dtype=float)
        day_net_revenue_qh_eur = np.zeros(len(group_idx), dtype=float)
        day_soc_trace = np.full(len(group_idx), soc_day_start, dtype=float)

        executed = False
        selected_charge_abs_idx = []
        selected_discharge_abs_idx = []

        charged_input_mwh_total = 0.0
        charged_stored_mwh_total = 0.0
        discharged_mwh_total = 0.0
        charge_cost_eur_total = 0.0
        sale_revenue_eur_total = 0.0
        cycle_cost_eur_total = 0.0
        net_revenue_eur_total = 0.0

        if best_trade["execute"]:
            charge_rel_indices = [int(i) for i in best_trade["charge_indices"]]
            discharge_rel_indices = [int(i) for i in best_trade["discharge_indices"]]

            selected_charge_abs_idx = [int(group_idx[i]) for i in charge_rel_indices]
            selected_discharge_abs_idx = [int(group_idx[i]) for i in discharge_rel_indices]

            soc_working = soc_day_start

            for rel_idx in charge_rel_indices:
                available_capacity_mwh = max(inputs.batt_energy_mwh - soc_working, 0.0)
                stored_this_qh = min(limits["stored_per_qh_mwh"], available_capacity_mwh)
                if stored_this_qh <= 1e-9:
                    continue

                input_this_qh = stored_this_qh / max(inputs.eta_charge, 1e-12)
                day_charge_qh_mwh[rel_idx] = input_this_qh
                day_charge_cost_qh_eur[rel_idx] = input_this_qh * charge_day[rel_idx]
                day_net_revenue_qh_eur[rel_idx] -= day_charge_cost_qh_eur[rel_idx]

                charged_input_mwh_total += input_this_qh
                charged_stored_mwh_total += stored_this_qh
                charge_cost_eur_total += day_charge_cost_qh_eur[rel_idx]

                soc_working += stored_this_qh
                day_soc_trace[rel_idx:] = soc_working

            for rel_idx in discharge_rel_indices:
                max_output_by_power = limits["output_per_qh_mwh"]
                max_output_by_export = inputs.grid_export_limit_mw * QH_DT_HOURS
                max_output_by_soc = soc_working * inputs.eta_discharge

                discharge_this_qh = min(max_output_by_power, max_output_by_export, max_output_by_soc)
                if discharge_this_qh <= 1e-9:
                    continue

                soc_removed_this_qh = discharge_this_qh / max(inputs.eta_discharge, 1e-12)

                day_discharge_qh_mwh[rel_idx] = discharge_this_qh
                day_sale_revenue_qh_eur[rel_idx] = discharge_this_qh * discharge_day[rel_idx]
                day_cycle_cost_qh_eur[rel_idx] = soc_removed_this_qh * inputs.afrr_cycle_cost_eur_per_mwh
                day_net_revenue_qh_eur[rel_idx] += day_sale_revenue_qh_eur[rel_idx] - day_cycle_cost_qh_eur[rel_idx]

                discharged_mwh_total += discharge_this_qh
                sale_revenue_eur_total += day_sale_revenue_qh_eur[rel_idx]
                cycle_cost_eur_total += day_cycle_cost_qh_eur[rel_idx]

                soc_working -= soc_removed_this_qh
                day_soc_trace[rel_idx:] = soc_working

            net_revenue_eur_total = sale_revenue_eur_total - charge_cost_eur_total - cycle_cost_eur_total

            if charged_input_mwh_total > 1e-9 and discharged_mwh_total > 1e-9 and net_revenue_eur_total >= 0.0:
                executed = True
                soc_current = soc_working

                afrr_charge_qh_mwh[group_idx] = day_charge_qh_mwh
                afrr_discharge_qh_mwh[group_idx] = day_discharge_qh_mwh
                afrr_charge_cost_qh_eur[group_idx] = day_charge_cost_qh_eur
                afrr_sale_revenue_qh_eur[group_idx] = day_sale_revenue_qh_eur
                afrr_cycle_cost_qh_eur[group_idx] = day_cycle_cost_qh_eur
                afrr_net_revenue_qh_eur[group_idx] = day_net_revenue_qh_eur
                afrr_soc_qh[group_idx] = day_soc_trace
            else:
                afrr_soc_qh[group_idx] = soc_day_start
                selected_charge_abs_idx = []
                selected_discharge_abs_idx = []
        else:
            afrr_soc_qh[group_idx] = soc_day_start

        daily_logs.append({
            "day": pd.to_datetime(day),
            "executed": executed,
            "charge_qh_indices": selected_charge_abs_idx,
            "discharge_qh_indices": selected_discharge_abs_idx,
            "charge_times": [idx_qh[i] for i in selected_charge_abs_idx],
            "discharge_times": [idx_qh[i] for i in selected_discharge_abs_idx],
            "avg_charge_price_eur_per_mwh": best_trade.get("avg_charge_price", np.nan),
            "avg_discharge_price_eur_per_mwh": best_trade.get("avg_discharge_price", np.nan),
            "expected_net_spread_eur_per_mwh": best_trade.get("expected_net_spread_eur_per_mwh", np.nan),
            "charged_input_mwh": charged_input_mwh_total,
            "charged_stored_mwh": charged_stored_mwh_total,
            "discharged_mwh": discharged_mwh_total,
            "charge_cost_eur": charge_cost_eur_total,
            "sale_revenue_eur": sale_revenue_eur_total,
            "cycle_cost_eur": cycle_cost_eur_total,
            "net_revenue_eur": net_revenue_eur_total,
            "reason": best_trade.get("reason", "OK"),
        })

    return {
        "afrr_charge_qh_mwh": afrr_charge_qh_mwh,
        "afrr_discharge_qh_mwh": afrr_discharge_qh_mwh,
        "afrr_soc_qh": afrr_soc_qh,
        "afrr_charge_cost_qh_eur": afrr_charge_cost_qh_eur,
        "afrr_sale_revenue_qh_eur": afrr_sale_revenue_qh_eur,
        "afrr_cycle_cost_qh_eur": afrr_cycle_cost_qh_eur,
        "afrr_net_revenue_qh_eur": afrr_net_revenue_qh_eur,
        "afrr_daily_log": pd.DataFrame(daily_logs),
    }


def reconcile_wholesale_afrr_dispatch_qh(
    result_hourly: Dict[str, np.ndarray],
    afrr_result: Dict[str, np.ndarray],
    inputs: SimulationInputs,
) -> Dict[str, np.ndarray]:
    idx_qh = build_quarter_hour_index(DEFAULT_YEAR)

    pv_direct_qh = np.repeat(np.asarray(result_hourly["pv_direct"], dtype=float) / QH_PER_HOUR, QH_PER_HOUR)
    wholesale_pv_to_batt_qh = np.repeat(np.asarray(result_hourly["pv_to_batt"], dtype=float) / QH_PER_HOUR, QH_PER_HOUR)
    wholesale_grid_charge_qh = np.repeat(np.asarray(result_hourly["grid_charge"], dtype=float) / QH_PER_HOUR, QH_PER_HOUR)
    wholesale_discharge_qh = np.repeat(np.asarray(result_hourly["discharge"], dtype=float) / QH_PER_HOUR, QH_PER_HOUR)

    batt_sell_price_qh = np.repeat(np.asarray(inputs.batt_sell_price, dtype=float), QH_PER_HOUR)
    grid_buy_price_qh = np.repeat(np.asarray(inputs.grid_buy_price, dtype=float), QH_PER_HOUR)

    afrr_charge_qh = np.asarray(afrr_result["afrr_charge_qh_mwh"], dtype=float).copy()
    afrr_discharge_qh = np.asarray(afrr_result["afrr_discharge_qh_mwh"], dtype=float).copy()
    afrr_charge_price_qh = np.asarray(inputs.afrr_charge_price_qh, dtype=float)
    afrr_discharge_price_qh = np.asarray(inputs.afrr_discharge_price_qh, dtype=float)

    corrected_wholesale_pv_to_batt_qh = wholesale_pv_to_batt_qh.copy()
    corrected_wholesale_grid_charge_qh = wholesale_grid_charge_qh.copy()
    corrected_wholesale_discharge_qh = wholesale_discharge_qh.copy()
    corrected_afrr_charge_qh = afrr_charge_qh.copy()
    corrected_afrr_discharge_qh = afrr_discharge_qh.copy()

    selected_discharge_channel_qh = np.full(QH_PER_YEAR, "none", dtype=object)
    selected_discharge_price_qh = np.full(QH_PER_YEAR, np.nan, dtype=float)

    export_limit_qh_mwh = inputs.grid_export_limit_mw * QH_DT_HOURS

    for t in range(QH_PER_YEAR):
        w_dis = corrected_wholesale_discharge_qh[t]
        a_dis = corrected_afrr_discharge_qh[t]

        w_price = batt_sell_price_qh[t] if w_dis > 1e-12 else -1e30
        a_price = afrr_discharge_price_qh[t] if a_dis > 1e-12 else -1e30

        if w_dis > 1e-12 and a_dis > 1e-12:
            if w_price >= a_price:
                corrected_afrr_discharge_qh[t] = 0.0
                selected_discharge_channel_qh[t] = "wholesale"
                selected_discharge_price_qh[t] = w_price
            else:
                corrected_wholesale_discharge_qh[t] = 0.0
                selected_discharge_channel_qh[t] = "afrr"
                selected_discharge_price_qh[t] = a_price
        elif w_dis > 1e-12:
            selected_discharge_channel_qh[t] = "wholesale"
            selected_discharge_price_qh[t] = w_price
        elif a_dis > 1e-12:
            selected_discharge_channel_qh[t] = "afrr"
            selected_discharge_price_qh[t] = a_price

        total_selected_discharge_qh = corrected_wholesale_discharge_qh[t] + corrected_afrr_discharge_qh[t]
        if total_selected_discharge_qh > 1e-12:
            corrected_wholesale_pv_to_batt_qh[t] = 0.0
            corrected_wholesale_grid_charge_qh[t] = 0.0
            corrected_afrr_charge_qh[t] = 0.0

        total_selected_discharge_qh = corrected_wholesale_discharge_qh[t] + corrected_afrr_discharge_qh[t]
        export_room_qh = max(export_limit_qh_mwh - pv_direct_qh[t], 0.0)

        if total_selected_discharge_qh > export_room_qh + 1e-12:
            if corrected_wholesale_discharge_qh[t] > 1e-12:
                corrected_wholesale_discharge_qh[t] = min(corrected_wholesale_discharge_qh[t], export_room_qh)
                corrected_afrr_discharge_qh[t] = 0.0
                if corrected_wholesale_discharge_qh[t] <= 1e-12:
                    selected_discharge_channel_qh[t] = "none"
                    selected_discharge_price_qh[t] = np.nan
            elif corrected_afrr_discharge_qh[t] > 1e-12:
                corrected_afrr_discharge_qh[t] = min(corrected_afrr_discharge_qh[t], export_room_qh)
                corrected_wholesale_discharge_qh[t] = 0.0
                if corrected_afrr_discharge_qh[t] <= 1e-12:
                    selected_discharge_channel_qh[t] = "none"
                    selected_discharge_price_qh[t] = np.nan

    corrected_wholesale_batt_sale_revenue_qh = corrected_wholesale_discharge_qh * batt_sell_price_qh
    corrected_wholesale_grid_charge_cost_qh = corrected_wholesale_grid_charge_qh * grid_buy_price_qh

    corrected_afrr_charge_cost_qh = corrected_afrr_charge_qh * afrr_charge_price_qh
    corrected_afrr_sale_revenue_qh = corrected_afrr_discharge_qh * afrr_discharge_price_qh
    corrected_afrr_cycle_cost_qh = (corrected_afrr_discharge_qh / max(inputs.eta_discharge, 1e-12)) * inputs.afrr_cycle_cost_eur_per_mwh
    corrected_afrr_net_revenue_qh = corrected_afrr_sale_revenue_qh - corrected_afrr_charge_cost_qh - corrected_afrr_cycle_cost_qh

    charge_to_soc_qh = (
        corrected_wholesale_pv_to_batt_qh
        + corrected_wholesale_grid_charge_qh
        + corrected_afrr_charge_qh
    ) * inputs.eta_charge

    discharge_from_soc_qh = (
        corrected_wholesale_discharge_qh
        + corrected_afrr_discharge_qh
    ) / max(inputs.eta_discharge, 1e-12)

    combined_soc_qh = np.zeros(QH_PER_YEAR + 1, dtype=float)
    combined_soc_qh[0] = float(inputs.initial_soc_mwh)

    for t in range(QH_PER_YEAR):
        soc_next = combined_soc_qh[t] + charge_to_soc_qh[t] - discharge_from_soc_qh[t]
        combined_soc_qh[t + 1] = min(max(soc_next, 0.0), inputs.batt_energy_mwh)

    def reshape_sum(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=float).reshape(HOURS_PER_YEAR, QH_PER_HOUR).sum(axis=1)

    def reshape_last(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=float).reshape(HOURS_PER_YEAR, QH_PER_HOUR)[:, -1]

    return {
        "datetime_qh": idx_qh,
        "wholesale_pv_to_batt_qh_mwh": corrected_wholesale_pv_to_batt_qh,
        "wholesale_grid_charge_qh_mwh": corrected_wholesale_grid_charge_qh,
        "wholesale_discharge_qh_mwh": corrected_wholesale_discharge_qh,
        "wholesale_batt_sale_revenue_qh_eur": corrected_wholesale_batt_sale_revenue_qh,
        "wholesale_grid_charge_cost_qh_eur": corrected_wholesale_grid_charge_cost_qh,
        "afrr_charge_qh_mwh": corrected_afrr_charge_qh,
        "afrr_discharge_qh_mwh": corrected_afrr_discharge_qh,
        "afrr_charge_cost_qh_eur": corrected_afrr_charge_cost_qh,
        "afrr_sale_revenue_qh_eur": corrected_afrr_sale_revenue_qh,
        "afrr_cycle_cost_qh_eur": corrected_afrr_cycle_cost_qh,
        "afrr_net_revenue_qh_eur": corrected_afrr_net_revenue_qh,
        "selected_discharge_channel_qh": selected_discharge_channel_qh,
        "selected_discharge_price_qh": selected_discharge_price_qh,
        "combined_charge_to_soc_qh_mwh": charge_to_soc_qh,
        "combined_discharge_from_soc_qh_mwh": discharge_from_soc_qh,
        "combined_soc_qh": combined_soc_qh,
        "combined_soc_hourly_end_mwh": reshape_last(combined_soc_qh[1:]),
        "wholesale_pv_to_batt_hourly_mwh": reshape_sum(corrected_wholesale_pv_to_batt_qh),
        "wholesale_grid_charge_hourly_mwh": reshape_sum(corrected_wholesale_grid_charge_qh),
        "wholesale_discharge_hourly_mwh": reshape_sum(corrected_wholesale_discharge_qh),
        "wholesale_batt_sale_revenue_hourly_eur": reshape_sum(corrected_wholesale_batt_sale_revenue_qh),
        "wholesale_grid_charge_cost_hourly_eur": reshape_sum(corrected_wholesale_grid_charge_cost_qh),
        "afrr_charge_hourly_mwh": reshape_sum(corrected_afrr_charge_qh),
        "afrr_discharge_hourly_mwh": reshape_sum(corrected_afrr_discharge_qh),
        "afrr_charge_cost_hourly_eur": reshape_sum(corrected_afrr_charge_cost_qh),
        "afrr_sale_revenue_hourly_eur": reshape_sum(corrected_afrr_sale_revenue_qh),
        "afrr_cycle_cost_hourly_eur": reshape_sum(corrected_afrr_cycle_cost_qh),
        "afrr_net_revenue_hourly_eur": reshape_sum(corrected_afrr_net_revenue_qh),
    }


def build_final_result_after_market_arbitration(
    base_result: Dict[str, np.ndarray],
    reconciliation: Dict[str, np.ndarray],
    inputs: SimulationInputs,
) -> Dict[str, np.ndarray]:
    final = dict(base_result)

    final["pv_to_batt"] = reconciliation["wholesale_pv_to_batt_hourly_mwh"]
    final["grid_charge"] = reconciliation["wholesale_grid_charge_hourly_mwh"]
    final["discharge"] = reconciliation["wholesale_discharge_hourly_mwh"]
    final["batt_sale_revenue"] = reconciliation["wholesale_batt_sale_revenue_hourly_eur"]
    final["grid_charge_cost"] = reconciliation["wholesale_grid_charge_cost_hourly_eur"]

    total_batt_sale_revenue = float(final["batt_sale_revenue"].sum())
    total_grid_charge_cost = float(final["grid_charge_cost"].sum())
    total_direct_pv_revenue = float(final["pv_direct_revenue"].sum())
    nightly_revenue_total = float(final["nightly_revenue_total"][0])

    final["total_batt_sale_revenue"] = np.array([total_batt_sale_revenue])
    final["total_grid_charge_cost"] = np.array([total_grid_charge_cost])
    final["energy_shifted_mwh"] = np.array([float(final["discharge"].sum())])
    final["energy_sold_total_mwh"] = np.array([float(final["pv_direct"].sum() + final["discharge"].sum())])
    final["equivalent_cycles"] = np.array([float(final["discharge"].sum() / max(inputs.batt_energy_mwh, 1e-12))])

    final["total_revenue"] = np.array([
        total_direct_pv_revenue + total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total
    ])

    final["afrr_charge_hourly_mwh"] = reconciliation["afrr_charge_hourly_mwh"]
    final["afrr_discharge_hourly_mwh"] = reconciliation["afrr_discharge_hourly_mwh"]
    final["afrr_charge_cost_hourly_eur"] = reconciliation["afrr_charge_cost_hourly_eur"]
    final["afrr_sale_revenue_hourly_eur"] = reconciliation["afrr_sale_revenue_hourly_eur"]
    final["afrr_cycle_cost_hourly_eur"] = reconciliation["afrr_cycle_cost_hourly_eur"]
    final["afrr_net_revenue_hourly_eur"] = reconciliation["afrr_net_revenue_hourly_eur"]

    final["total_afrr_charge_cost_eur"] = np.array([float(reconciliation["afrr_charge_cost_hourly_eur"].sum())])
    final["total_afrr_sale_revenue_eur"] = np.array([float(reconciliation["afrr_sale_revenue_hourly_eur"].sum())])
    final["total_afrr_cycle_cost_eur"] = np.array([float(reconciliation["afrr_cycle_cost_hourly_eur"].sum())])
    final["total_afrr_net_revenue_eur"] = np.array([float(reconciliation["afrr_net_revenue_hourly_eur"].sum())])

    final["total_battery_revenue_including_afrr_eur"] = np.array([
        total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total + float(reconciliation["afrr_net_revenue_hourly_eur"].sum())
    ])

    final["total_revenue_including_afrr_eur"] = np.array([
        total_direct_pv_revenue + total_batt_sale_revenue - total_grid_charge_cost + nightly_revenue_total + float(reconciliation["afrr_net_revenue_hourly_eur"].sum())
    ])

    return final


def build_summary_table(
    result: Dict[str, np.ndarray],
    pv_stats: Dict[str, float],
    pure_pv_benchmark: Dict[str, np.ndarray],
    pv_dc_mw: float,
    batt_power_mw: float,
    pv_capture_rate_pct: float,
    bess_capture_rate_pct: float,
    curtailment_outputs: Dict[str, np.ndarray],
) -> pd.DataFrame:
    pv_revenue = float(result["total_direct_pv_revenue"][0])

    bess_revenue_base = (
        float(result["total_batt_sale_revenue"][0])
        - float(result["total_grid_charge_cost"][0])
        + float(result["nightly_revenue_total"][0])
    )

    afrr_net_revenue = float(result["total_afrr_net_revenue_eur"][0]) if "total_afrr_net_revenue_eur" in result else 0.0
    afrr_sale_revenue = float(result["total_afrr_sale_revenue_eur"][0]) if "total_afrr_sale_revenue_eur" in result else 0.0
    afrr_charge_cost = float(result["total_afrr_charge_cost_eur"][0]) if "total_afrr_charge_cost_eur" in result else 0.0
    afrr_cycle_cost = float(result["total_afrr_cycle_cost_eur"][0]) if "total_afrr_cycle_cost_eur" in result else 0.0

    bess_revenue_total = bess_revenue_base + afrr_net_revenue
    total_revenue = float(result["total_revenue_including_afrr_eur"][0]) if "total_revenue_including_afrr_eur" in result else float(result["total_revenue"][0])

    pure_pv_revenue = float(pure_pv_benchmark["total_pv_only_revenue_eur"][0])
    hybrid_added_value = total_revenue - pure_pv_revenue

    pv_rev_keur_per_mw = pv_revenue / max(pv_dc_mw, 1e-12) / 1000.0
    bess_rev_keur_per_mw = bess_revenue_total / max(batt_power_mw, 1e-12) / 1000.0

    pv_sold_mwh = float(result["pv_direct_sold_mwh"][0])
    bess_sold_mwh = float(result["energy_shifted_mwh"][0])

    afrr_discharged_mwh = float(np.sum(result["afrr_discharge_hourly_mwh"])) if "afrr_discharge_hourly_mwh" in result else 0.0
    bess_total_discharged_mwh = bess_sold_mwh + afrr_discharged_mwh

    pv_rev_eur_per_mwh = pv_revenue / max(pv_sold_mwh, 1e-12)
    bess_rev_eur_per_mwh = bess_revenue_total / max(bess_total_discharged_mwh, 1e-12)

    tso_dso_curtailed = float(np.sum(curtailment_outputs["tso_dso_curtailed_mwh"]))
    self_curtailed = float(np.sum(curtailment_outputs["self_curtailed_mwh"]))
    candidate_curtailed = float(np.sum(curtailment_outputs["pv_curtailment_candidate_mwh"]))
    recovered_to_battery = float(np.sum(curtailment_outputs["pv_curtailed_to_battery_mwh_actual"]))
    residual_lost = float(np.sum(curtailment_outputs["pv_curtailed_residual_lost_mwh"]))

    rows = [
        ("PV Capture Rate", pv_capture_rate_pct, "%"),
        ("BESS Capture Rate", bess_capture_rate_pct, "%"),
        ("Revenu total", total_revenue, "EUR"),
        ("Revenu PV-only Project", pure_pv_revenue, "EUR"),
        ("Valeur ajoutée de l'hybridation vs PV-only", hybrid_added_value, "EUR"),
        ("Revenu PV direct", pv_revenue, "EUR"),
        ("Revenu batterie wholesale", float(result["total_batt_sale_revenue"][0]), "EUR"),
        ("Coût charge réseau wholesale", float(result["total_grid_charge_cost"][0]), "EUR"),
        ("Revenu services système de nuit", float(result["nightly_revenue_total"][0]), "EUR"),
        ("Revenu brut aFRR", afrr_sale_revenue, "EUR"),
        ("Cashflow charge aFRR", afrr_charge_cost, "EUR"),
        ("Coût cycle aFRR", afrr_cycle_cost, "EUR"),
        ("Revenu net aFRR", afrr_net_revenue, "EUR"),
        ("TSO/DSO curtailed energy", tso_dso_curtailed, "MWh"),
        ("Self-curtailed energy", self_curtailed, "MWh"),
        ("Total curtailed PV candidate energy", candidate_curtailed, "MWh"),
        ("Curtailed PV recovered by battery", recovered_to_battery, "MWh"),
        ("Residual curtailed PV energy lost", residual_lost, "MWh"),
        ("Revenu PV spécifique", pv_rev_keur_per_mw, "kEUR/MW"),
        ("Revenu BESS spécifique", bess_rev_keur_per_mw, "kEUR/MW"),
        ("Revenu PV spécifique énergie", pv_rev_eur_per_mwh, "€/MWh"),
        ("Revenu BESS spécifique énergie", bess_rev_eur_per_mwh, "€/MWh"),
        ("Énergie totale vendue", float(result["energy_sold_total_mwh"][0]) + afrr_discharged_mwh, "MWh"),
        ("Énergie shiftée wholesale", bess_sold_mwh, "MWh"),
        ("Énergie déchargée aFRR", afrr_discharged_mwh, "MWh"),
        ("Énergie PV vendue directement", pv_sold_mwh, "MWh"),
        ("Cycles équivalents batterie", float(result["equivalent_cycles"][0]), "cycles/an"),
        ("Production PV théorique brute", float(pv_stats["annual_dc_mwh"]), "MWh"),
        ("Production PV nette valorisable", float(pv_stats["annual_net_mwh"]), "MWh"),
        ("Énergie PV perdue (pertes + disponibilité)", float(pv_stats["annual_losses_mwh"]), "MWh"),
    ]
    return pd.DataFrame(rows, columns=["Indicateur", "Valeur", "Unité"])


def monthly_dataframe(
    result: Dict[str, np.ndarray],
    pure_pv_benchmark: Dict[str, np.ndarray],
    pv_dc_mw: float,
    batt_power_mw: float,
    curtailment_outputs: Dict[str, np.ndarray],
) -> pd.DataFrame:
    idx = pd.date_range(f"{DEFAULT_YEAR}-01-01 00:00:00", periods=HOURS_PER_YEAR, freq="h")

    df = pd.DataFrame({
        "datetime": idx,
        "pv_direct_revenue": result["pv_direct_revenue"],
        "batt_sale_revenue": result["batt_sale_revenue"],
        "grid_charge_cost": result["grid_charge_cost"],
        "pv_direct_mwh": result["pv_direct"],
        "shifted_mwh": result["discharge"],
        "grid_charge_mwh": result["grid_charge"],
        "pv_to_batt_mwh": result["pv_to_batt"],
        "pv_curtailed_to_battery_mwh_actual": curtailment_outputs["pv_curtailed_to_battery_mwh_actual"],
        "pv_curtailment_candidate_mwh": curtailment_outputs["pv_curtailment_candidate_mwh"],
        "pv_curtailed_residual_lost_mwh": curtailment_outputs["pv_curtailed_residual_lost_mwh"],
        "pv_only_direct_mwh": pure_pv_benchmark["pv_only_direct_mwh"],
        "pv_only_revenue": pure_pv_benchmark["pv_only_revenue_eur"],
        "afrr_charge_mwh": result["afrr_charge_hourly_mwh"] if "afrr_charge_hourly_mwh" in result else np.zeros(HOURS_PER_YEAR),
        "afrr_discharge_mwh": result["afrr_discharge_hourly_mwh"] if "afrr_discharge_hourly_mwh" in result else np.zeros(HOURS_PER_YEAR),
        "afrr_charge_cost": result["afrr_charge_cost_hourly_eur"] if "afrr_charge_cost_hourly_eur" in result else np.zeros(HOURS_PER_YEAR),
        "afrr_sale_revenue": result["afrr_sale_revenue_hourly_eur"] if "afrr_sale_revenue_hourly_eur" in result else np.zeros(HOURS_PER_YEAR),
        "afrr_cycle_cost": result["afrr_cycle_cost_hourly_eur"] if "afrr_cycle_cost_hourly_eur" in result else np.zeros(HOURS_PER_YEAR),
        "afrr_net_revenue": result["afrr_net_revenue_hourly_eur"] if "afrr_net_revenue_hourly_eur" in result else np.zeros(HOURS_PER_YEAR),
    })

    df["month"] = df["datetime"].dt.strftime("%Y-%m")
    monthly = df.groupby("month", as_index=False).sum(numeric_only=True)

    monthly["bess_net_revenue"] = monthly["batt_sale_revenue"] - monthly["grid_charge_cost"] + monthly["afrr_net_revenue"]
    monthly["net_revenue"] = monthly["pv_direct_revenue"] + monthly["bess_net_revenue"]

    monthly["pv_revenue_keur_per_mw"] = monthly["pv_direct_revenue"] / max(pv_dc_mw, 1e-12) / 1000.0
    monthly["bess_revenue_keur_per_mw"] = monthly["bess_net_revenue"] / max(batt_power_mw, 1e-12) / 1000.0

    monthly["pv_revenue_eur_per_mwh"] = monthly["pv_direct_revenue"] / monthly["pv_direct_mwh"].clip(lower=1e-12)
    monthly["bess_total_discharged_mwh"] = monthly["shifted_mwh"] + monthly["afrr_discharge_mwh"]
    monthly["bess_revenue_eur_per_mwh"] = monthly["bess_net_revenue"] / monthly["bess_total_discharged_mwh"].clip(lower=1e-12)

    return monthly


def build_inputs_dataframe(inputs: SimulationInputs) -> pd.DataFrame:
    rows = [
        ("batt_power_mw", inputs.batt_power_mw),
        ("batt_energy_mwh", inputs.batt_energy_mwh),
        ("pv_dc_mw", inputs.pv_dc_mw),
        ("productible_kwh_per_kwp", inputs.productible_kwh_per_kwp),
        ("pv_losses_pct", inputs.pv_losses_pct),
        ("plant_availability_pct", inputs.plant_availability_pct),
        ("eta_charge", inputs.eta_charge),
        ("eta_discharge", inputs.eta_discharge),
        ("nightly_bess_revenue_eur", inputs.nightly_bess_revenue_eur),
        ("soc_steps", inputs.soc_steps),
        ("initial_soc_mwh", inputs.initial_soc_mwh),
        ("final_soc_mwh", inputs.final_soc_mwh),
        ("grid_export_limit_mw", inputs.grid_export_limit_mw),
        ("cycle_cost_eur_per_mwh", inputs.cycle_cost_eur_per_mwh),
        ("charge_quantile", inputs.charge_quantile),
        ("discharge_quantile", inputs.discharge_quantile),
        ("max_cycles_per_day", inputs.max_cycles_per_day),
        ("min_spread_arbitrage_eur_per_mwh", inputs.min_spread_arbitrage_eur_per_mwh),
        ("pv_capture_rate_pct", inputs.pv_capture_rate_pct),
        ("bess_capture_rate_pct", inputs.bess_capture_rate_pct),
        ("enable_afrr", inputs.enable_afrr),
        ("afrr_min_spread_eur_per_mwh", inputs.afrr_min_spread_eur_per_mwh),
        ("afrr_cycle_cost_eur_per_mwh", inputs.afrr_cycle_cost_eur_per_mwh),
        ("afrr_max_events_per_day", inputs.afrr_max_events_per_day),
        ("afrr_night_start_hour", inputs.afrr_night_start_hour),
        ("afrr_night_end_hour", inputs.afrr_night_end_hour),
        ("afrr_pv_zero_tolerance_mwh", inputs.afrr_pv_zero_tolerance_mwh),
        ("afrr_n_qh_per_side", inputs.afrr_n_qh_per_side),
        ("enable_tso_dso_curtailment", inputs.enable_tso_dso_curtailment),
        ("enable_self_curtailment", inputs.enable_self_curtailment),
        ("curtailment_threshold_eur_per_mwh", inputs.curtailment_threshold_eur_per_mwh),
        ("pv_commercial_structure", inputs.pv_commercial_structure),
        ("cfd_price_eur_per_mwh", inputs.cfd_price_eur_per_mwh),
        ("negative_price_rule", inputs.negative_price_rule),
        ("consecutive_negative_hours_limit", inputs.consecutive_negative_hours_limit),
        ("ppa_price_eur_per_mwh", inputs.ppa_price_eur_per_mwh),
        ("charge_battery_if_curtailment", inputs.charge_battery_if_curtailment),
        ("enable_cfd", inputs.enable_cfd),
        ("cfd_price_standalone_eur_per_mwh", inputs.cfd_price_standalone_eur_per_mwh),
        ("enable_ppa", inputs.enable_ppa),
        ("ppa_price_standalone_eur_per_mwh", inputs.ppa_price_standalone_eur_per_mwh),
        ("bess_degradation_curve_pct", "" if inputs.bess_degradation_curve_pct is None else list(inputs.bess_degradation_curve_pct)),
        ("degraded_bess_energy_by_year_mwh", "" if inputs.degraded_bess_energy_by_year_mwh is None else list(inputs.degraded_bess_energy_by_year_mwh)),
        ("project_lifetime_years", inputs.project_lifetime_years),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


def to_excel_bytes(
    inputs_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    afrr_qh_df: pd.DataFrame | None = None,
    afrr_daily_log_df: pd.DataFrame | None = None,
    bess_degradation_df: pd.DataFrame | None = None,
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        inputs_df.to_excel(writer, sheet_name="Inputs", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        monthly_df.to_excel(writer, sheet_name="Monthly", index=False)
        hourly_df.to_excel(writer, sheet_name="Hourly", index=False)

        if afrr_qh_df is not None:
            afrr_qh_df.to_excel(writer, sheet_name="aFRR_QH", index=False)

        if afrr_daily_log_df is not None:
            afrr_daily_log_df.to_excel(writer, sheet_name="aFRR_Daily_Log", index=False)

        if bess_degradation_df is not None:
            bess_degradation_df.to_excel(writer, sheet_name="BESS_Degradation", index=False)

    return output.getvalue()


def app():
    st.set_page_config(page_title="Évaluation revenus projet hybride PV + BESS", layout="wide")
    st.title("Évaluation des revenus d'un projet hybride PV + batterie")
    st.caption("Simulation 8760h avec optimisation économique annuelle de la batterie + arbitrage aFRR quart-horaire de nuit.")

    with st.expander("Hypothèses structurantes", expanded=False):
        st.markdown(
            """
            - Simulation **horaire sur 8760h** pour le cœur du dispatch PV + BESS.
            - La batterie peut **charger depuis le PV et/ou depuis le réseau**.
            - Le moteur choisit la meilleure valorisation économique entre vente immédiate du PV, stockage PV et charge réseau.
            - Les **revenus de services système la nuit** sont ajoutés comme un **revenu fixe par nuit**, sans contrainte de capacité ni de SOC.
            - L'optimisation principale utilise une **programmation dynamique discrétisée sur le SOC**.
            - Une couche séparée **aFRR quart-horaire** peut être ajoutée la nuit sans production PV.
            - La curtailment PV peut être:
              1. imposée par TSO/DSO
              2. auto-courtailment selon structure commerciale
            - Option supplémentaire: **Charge Battery if Curtailment**
              pour récupérer une partie de l'énergie autrement curtailed dans la batterie.
            """
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        batt_power_mw = st.number_input("Puissance batterie utile (MW)", min_value=0.0, value=50.0, step=1.0)
        batt_energy_mwh = st.number_input("Capacité batterie utile (MWh)", min_value=0.0, value=100.0, step=1.0)
        pv_dc_mw = st.number_input("Puissance PV DC (MWc)", min_value=0.0, value=100.0, step=1.0)
        productible = st.number_input("Productible PV (kWh/kWc/an)", min_value=0.0, value=1200.0, step=10.0)
        grid_export_limit_mw = st.number_input("Limite injection réseau (MW)", min_value=0.0, value=100.0, step=1.0)
        cycle_cost = st.number_input("Coût de cycle batterie (EUR/MWh)", value=5.0)
        min_spread_arbitrage = st.number_input("Minimum Spread for Arbitrage (EUR/MWh)", min_value=0.0, value=10.0, step=1.0)
        charge_quantile = st.slider("Quantile charge (%)", 0, 50, 20)
        discharge_quantile = st.slider("Quantile décharge (%)", 0, 100, 80)
        max_cycles = st.number_input("Cycles max / jour", min_value=0.0, value=1.0, step=0.1)

    with col2:
        pv_losses_pct = st.number_input("Pertes système PV (%)", min_value=0.0, max_value=100.0, value=8.0, step=0.5)
        availability_pct = st.number_input("Disponibilité globale centrale (%)", min_value=0.0, max_value=100.0, value=98.0, step=0.1)
        eta_charge = st.number_input("Rendement de charge batterie (%)", min_value=1.0, max_value=100.0, value=95.0, step=0.5) / 100.0
        eta_discharge = st.number_input("Rendement de décharge batterie (%)", min_value=1.0, max_value=100.0, value=95.0, step=0.5) / 100.0
        pv_capture_rate_pct = st.number_input("PV Capture Rate (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)

    with col3:
        nightly_bess_revenue = st.number_input("Revenu services système nuit (EUR/nuit)", min_value=0.0, value=0.0, step=10.0)
        soc_steps = st.slider("Nombre de pas de SOC pour l'optimisation", min_value=21, max_value=201, value=101, step=10)
        initial_soc = st.number_input("SOC initial batterie (MWh)", min_value=0.0, value=0.0, step=1.0)
        final_soc = st.number_input("SOC final cible batterie (MWh)", min_value=0.0, value=0.0, step=1.0)
        bess_capture_rate_pct = st.number_input("BESS Capture Rate (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)

    st.subheader("Courbe solaire 8760h")
    solar_mode = st.radio("Source du profil solaire", ["Courbe standard France", "Upload CSV 8760"], horizontal=True)

    solar_upload = None
    uploaded_solar_is_relative = True
    if solar_mode == "Upload CSV 8760":
        solar_upload = st.file_uploader("Upload du profil solaire CSV (8760 lignes, première colonne numérique)", type=["csv"], key="solar_csv")
        uploaded_solar_is_relative = st.checkbox(
            "Le CSV uploadé est un profil relatif à normaliser sur le productible annuel (sinon : MWh nets horaires absolus)",
            value=True,
        )

    st.subheader("Prix PV")
    pv_price_mode = st.radio("Source du prix de vente du PV", ["Prix moyen annuel", "Upload CSV 8760"], horizontal=True)
    pv_price_value = None
    pv_price_upload = None
    if pv_price_mode == "Prix moyen annuel":
        pv_price_value = st.number_input("Prix moyen PV (EUR/MWh)", value=55.0, step=1.0)
    else:
        pv_price_upload = st.file_uploader("Upload prix PV CSV (8760 lignes)", type=["csv"], key="pv_price")

    st.subheader("Prix vente batterie")
    batt_sell_mode = st.radio("Source du prix de vente de l'énergie shiftée", ["Prix moyen annuel", "Upload CSV 8760"], horizontal=True)
    batt_sell_value = None
    batt_sell_upload = None
    if batt_sell_mode == "Prix moyen annuel":
        batt_sell_value = st.number_input("Prix moyen vente batterie (EUR/MWh)", value=90.0, step=1.0)
    else:
        batt_sell_upload = st.file_uploader("Upload prix vente batterie CSV (8760 lignes)", type=["csv"], key="batt_sell")

    st.subheader("Prix d'achat réseau pour charge batterie")
    grid_mode = st.radio("Source du prix d'achat réseau", ["Identique au prix vente batterie", "Prix moyen annuel", "Upload CSV 8760"], horizontal=True)
    grid_buy_value = None
    grid_buy_upload = None
    if grid_mode == "Prix moyen annuel":
        grid_buy_value = st.number_input("Prix moyen achat réseau (EUR/MWh)", value=55.0, step=1.0)
    elif grid_mode == "Upload CSV 8760":
        grid_buy_upload = st.file_uploader("Upload prix achat réseau CSV (8760 lignes)", type=["csv"], key="grid_buy")

    st.subheader("Curtailment")
    cur1, cur2, cur3 = st.columns(3)

    with cur1:
        tso_dso_curtailment = st.radio("TSO/DSO Curtailment", ["No", "Yes"], horizontal=True)
        tso_dso_upload = None
        if tso_dso_curtailment == "Yes":
            tso_dso_upload = st.file_uploader("Upload Annual Curtailment Curve Excel (12 monthly %)", type=["xlsx", "xls"], key="tso_dso_curve")

    with cur2:
        self_curtailment = st.radio("Self Curtailment", ["No", "Yes"], horizontal=True)
        curtailment_threshold = -1.0
        pv_structure = "Fully merchant"
        cfd_price = 0.0
        negative_price_rule = False
        consecutive_negative_hours_limit = 6
        ppa_price = 0.0

        if self_curtailment == "Yes":
            curtailment_threshold = st.number_input("Curtailment Threshold (EUR/MWh)", value=-1.0, step=1.0)
            pv_structure = st.radio("PV Commercial Structure", ["Fully merchant", "With CfD", "With PPA"], horizontal=False)

            if pv_structure == "With CfD":
                cfd_price = st.number_input("CfD Price (EUR/MWh)", value=50.0, step=1.0)
                negative_price_rule_str = st.radio("Negative Price Rule", ["No", "Yes"], horizontal=True)
                negative_price_rule = negative_price_rule_str == "Yes"
                if negative_price_rule:
                    consecutive_negative_hours_limit = int(st.number_input("Consecutive Negative Hours Limit", min_value=1, value=6, step=1))

            if pv_structure == "With PPA":
                ppa_price = st.number_input("PPA Price (EUR/MWh)", value=50.0, step=1.0)

    with cur3:
        charge_battery_if_curtailment = st.radio("Charge Battery if Curtailment", ["No", "Yes"], horizontal=True) == "Yes"
        
    st.subheader("Contrats PV et durée projet")

    contract_col1, contract_col2, contract_col3 = st.columns(3)

    with contract_col1:
        enable_cfd = st.radio("CfD", ["No", "Yes"], horizontal=True) == "Yes"
        cfd_price_standalone = 0.0
        if enable_cfd:
            cfd_price_standalone = st.number_input("CfD Price (€/MWh)", value=50.0, step=1.0)

    with contract_col2:
        enable_ppa = st.radio("PPA", ["No", "Yes"], horizontal=True) == "Yes"
        ppa_price_standalone = 0.0
        if enable_ppa:
            ppa_price_standalone = st.number_input("PPA Price (€/MWh)", value=50.0, step=1.0)

    with contract_col3:
        project_lifetime_years = int(
            st.number_input("Project Lifetime", min_value=1, value=1, step=1)
        )
        bess_degradation_upload = st.file_uploader(
            "BESS Degradation Curve",
            type=["xlsx", "xls"],
            key="bess_degradation_curve",
        )
        
    st.subheader("aFRR énergie (quart-horaire)")
    enable_afrr = st.checkbox("Activer l'arbitrage aFRR de nuit", value=False)

    afrr_charge_upload = None
    afrr_discharge_upload = None
    afrr_min_spread = 0.0
    afrr_cycle_cost = cycle_cost
    afrr_night_start_hour = 20
    afrr_night_end_hour = 8
    afrr_max_events_per_day = 1

    if enable_afrr:
        c_afrr1, c_afrr2, c_afrr3 = st.columns(3)

        with c_afrr1:
            afrr_charge_upload = st.file_uploader("Upload prix aFRR charge CSV (35040 lignes)", type=["csv"], key="afrr_charge")
            afrr_discharge_upload = st.file_uploader("Upload prix aFRR décharge CSV (35040 lignes)", type=["csv"], key="afrr_discharge")

        with c_afrr2:
            afrr_min_spread = st.number_input("Spread minimum aFRR net (EUR/MWh)", min_value=0.0, value=10.0, step=1.0)
            afrr_cycle_cost = st.number_input("Coût de cycle aFRR (EUR/MWh)", min_value=0.0, value=float(cycle_cost), step=1.0)

        with c_afrr3:
            afrr_night_start_hour = st.slider("Début nuit", 0, 23, 20)
            afrr_night_end_hour = st.slider("Fin nuit", 0, 23, 8)
            afrr_max_events_per_day = st.number_input("Nombre max d'événements aFRR / jour", min_value=1, value=1, step=1)

    st.markdown("---")
    run = st.button("Lancer la simulation", type="primary")

    if not run:
        return

    start_time = time.time()

    try:
        if batt_energy_mwh < batt_power_mw and batt_energy_mwh > 0:
            st.warning("Attention : la capacité batterie est inférieure à 1h de puissance. C'est possible, mais atypique.")
        if initial_soc > batt_energy_mwh:
            st.error("Le SOC initial ne peut pas dépasser la capacité batterie.")
            return
        if final_soc > batt_energy_mwh:
            st.error("Le SOC final ne peut pas dépasser la capacité batterie.")
            return
        if enable_cfd and enable_ppa:
            st.error("CfD et PPA ne peuvent pas être activés en même temps.")
            return

        try:
            bess_degradation_curve_pct, degraded_bess_energy_by_year_mwh, bess_degradation_df = read_bess_degradation_excel(
                bess_degradation_upload,
                project_lifetime_years,
                batt_energy_mwh,
            )
        except Exception as e:
            st.error(f"Erreur courbe de dégradation BESS: {e}")
            return
            
        # Base PV
        if solar_mode == "Courbe standard France":
            solar_relative = build_standard_france_solar_profile()
            base_pv_hourly_mwh, pv_stats = build_pv_generation_mwh(
                solar_relative, pv_dc_mw, productible, pv_losses_pct, availability_pct
            )
        else:
            if solar_upload is None:
                st.error("Merci d'uploader un CSV solaire 8760.")
                return

            uploaded = _read_single_column_csv(solar_upload)
            if uploaded_solar_is_relative:
                base_pv_hourly_mwh, pv_stats = build_pv_generation_mwh(
                    uploaded, pv_dc_mw, productible, pv_losses_pct, availability_pct
                )
            else:
                base_pv_hourly_mwh = np.maximum(uploaded, 0.0) * pv_dc_mw
                annual_net = float(base_pv_hourly_mwh.sum())
                annual_dc = float(pv_dc_mw * productible)
                pv_stats = {
                    "annual_dc_mwh": annual_dc,
                    "annual_net_mwh": annual_net,
                    "annual_losses_mwh": float(max(annual_dc - annual_net, 0.0)),
                }

            # Raw price curves
            if enable_cfd:
                pv_price_curve_raw = _make_flat_curve(cfd_price_standalone)
            elif enable_ppa:
                pv_price_curve_raw = _make_flat_curve(ppa_price_standalone)
            else:
                pv_price_curve_raw = _make_flat_curve(pv_price_value) if pv_price_mode == "Prix moyen annuel" else _read_single_column_csv(pv_price_upload)
            batt_sell_curve_raw = _make_flat_curve(batt_sell_value) if batt_sell_mode == "Prix moyen annuel" else _read_single_column_csv(batt_sell_upload)

        if grid_mode == "Identique au prix vente batterie":
            grid_buy_curve_raw = batt_sell_curve_raw.copy()
        elif grid_mode == "Prix moyen annuel":
            grid_buy_curve_raw = _make_flat_curve(grid_buy_value)
        else:
            grid_buy_curve_raw = _read_single_column_csv(grid_buy_upload)

        afrr_charge_curve_qh_raw = None
        afrr_discharge_curve_qh_raw = None
        if enable_afrr:
            if afrr_charge_upload is None or afrr_discharge_upload is None:
                st.error("Merci d'uploader les deux CSV aFRR quart-horaires.")
                return
            afrr_charge_curve_qh_raw = _read_single_column_csv_qh(afrr_charge_upload)
            afrr_discharge_curve_qh_raw = _read_single_column_csv_qh(afrr_discharge_upload)

        # Capture rates
        pv_capture_factor = pv_capture_rate_pct / 100.0
        bess_capture_factor = bess_capture_rate_pct / 100.0

        pv_spot_price_effective = pv_price_curve_raw * pv_capture_factor
        batt_sell_curve_effective = batt_sell_curve_raw * bess_capture_factor
        grid_buy_curve_effective = grid_buy_curve_raw * bess_capture_factor

        afrr_charge_curve_qh_effective = None
        afrr_discharge_curve_qh_effective = None
        if enable_afrr:
            afrr_charge_curve_qh_effective = afrr_charge_curve_qh_raw * bess_capture_factor
            afrr_discharge_curve_qh_effective = afrr_discharge_curve_qh_raw * bess_capture_factor

        # 1) TSO/DSO curtailment
        if tso_dso_curtailment == "Yes":
            if tso_dso_upload is None:
                st.error("Merci d'uploader la courbe annuelle de curtailment TSO/DSO.")
                return
            tso_dso_monthly_pct = read_monthly_curtailment_excel(tso_dso_upload)
            tso_out = apply_tso_dso_curtailment(base_pv_hourly_mwh, tso_dso_monthly_pct)
        else:
            tso_out = {
                "pv_after_tso_dso_mwh": base_pv_hourly_mwh.copy(),
                "tso_dso_curtailed_mwh": np.zeros(HOURS_PER_YEAR, dtype=float),
                "tso_dso_curtailment_flag": np.zeros(HOURS_PER_YEAR, dtype=int),
                "tso_dso_monthly_pct_hourly": np.zeros(HOURS_PER_YEAR, dtype=float),
            }
            tso_dso_monthly_pct = np.zeros(12, dtype=float)

        # 2) Self curtailment
        self_out = apply_self_curtailment(
            pv_hourly_mwh=tso_out["pv_after_tso_dso_mwh"],
            pv_spot_price_raw=pv_price_curve_raw,
            pv_spot_price_effective=pv_spot_price_effective,
            enable_self_curtailment=(self_curtailment == "Yes"),
            pv_commercial_structure=pv_structure,
            curtailment_threshold_eur_per_mwh=curtailment_threshold,
            cfd_price_eur_per_mwh=cfd_price,
            negative_price_rule=negative_price_rule,
            consecutive_negative_hours_limit=consecutive_negative_hours_limit,
            ppa_price_eur_per_mwh=ppa_price,
        )

        if enable_cfd or enable_ppa:
            self_out["pv_effective_price_eur_per_mwh"] = pv_spot_price_effective.copy()
            
        # Curtailment pipeline
        pv_after_tso_dso = tso_out["pv_after_tso_dso_mwh"]
        pv_after_self = self_out["pv_after_self_curtailment_mwh"]
        pv_curtailment_candidate_mwh = np.maximum(base_pv_hourly_mwh - pv_after_self, 0.0)

        if charge_battery_if_curtailment:
            curtailed_pv_recoverable_mwh = pv_curtailment_candidate_mwh.copy()
        else:
            curtailed_pv_recoverable_mwh = np.zeros(HOURS_PER_YEAR, dtype=float)

        pv_sellable_for_dispatch_mwh = pv_after_self.copy()
        pv_effective_price_for_revenue = self_out["pv_effective_price_eur_per_mwh"]

        # PV-only benchmark uses only sellable curtailed PV
        pure_pv_benchmark = build_pure_pv_benchmark(
            pv_generation_mwh=pv_sellable_for_dispatch_mwh,
            pv_price=pv_effective_price_for_revenue,
            grid_export_limit_mw=grid_export_limit_mw,
        )

        sim_inputs = SimulationInputs(
            batt_power_mw=batt_power_mw,
            batt_energy_mwh=batt_energy_mwh,
            pv_dc_mw=pv_dc_mw,
            productible_kwh_per_kwp=productible,
            pv_losses_pct=pv_losses_pct,
            plant_availability_pct=availability_pct,
            eta_charge=eta_charge,
            eta_discharge=eta_discharge,
            pv_price=pv_effective_price_for_revenue,
            batt_sell_price=batt_sell_curve_effective,
            grid_buy_price=grid_buy_curve_effective,
            solar_profile=pv_sellable_for_dispatch_mwh,
            curtailed_pv_recoverable_mwh=curtailed_pv_recoverable_mwh,
            nightly_bess_revenue_eur=nightly_bess_revenue,
            soc_steps=soc_steps,
            initial_soc_mwh=initial_soc,
            final_soc_mwh=final_soc,
            grid_export_limit_mw=grid_export_limit_mw,
            cycle_cost_eur_per_mwh=cycle_cost,
            charge_quantile=charge_quantile,
            discharge_quantile=discharge_quantile,
            max_cycles_per_day=max_cycles,
            min_spread_arbitrage_eur_per_mwh=min_spread_arbitrage,
            pv_capture_rate_pct=pv_capture_rate_pct,
            bess_capture_rate_pct=bess_capture_rate_pct,
            enable_afrr=enable_afrr,
            afrr_charge_price_qh=afrr_charge_curve_qh_effective,
            afrr_discharge_price_qh=afrr_discharge_curve_qh_effective,
            afrr_min_spread_eur_per_mwh=afrr_min_spread,
            afrr_cycle_cost_eur_per_mwh=afrr_cycle_cost,
            afrr_max_events_per_day=int(afrr_max_events_per_day),
            afrr_night_start_hour=int(afrr_night_start_hour),
            afrr_night_end_hour=int(afrr_night_end_hour),
            afrr_pv_zero_tolerance_mwh=PV_ZERO_TOLERANCE_MWH,
            afrr_n_qh_per_side=4,
            enable_tso_dso_curtailment=(tso_dso_curtailment == "Yes"),
            tso_dso_monthly_curtailment_pct=tso_dso_monthly_pct,
            enable_self_curtailment=(self_curtailment == "Yes"),
            curtailment_threshold_eur_per_mwh=curtailment_threshold,
            pv_commercial_structure=pv_structure,
            cfd_price_eur_per_mwh=cfd_price,
            negative_price_rule=negative_price_rule,
            consecutive_negative_hours_limit=consecutive_negative_hours_limit,
            ppa_price_eur_per_mwh=ppa_price,
            charge_battery_if_curtailment=charge_battery_if_curtailment,
            enable_cfd=enable_cfd,
            cfd_price_standalone_eur_per_mwh=cfd_price_standalone,
            enable_ppa=enable_ppa,
            ppa_price_standalone_eur_per_mwh=ppa_price_standalone,
            project_lifetime_years=project_lifetime_years,
            bess_degradation_curve_pct=bess_degradation_curve_pct,
            degraded_bess_energy_by_year_mwh=degraded_bess_energy_by_year_mwh,
        )

        inputs_df = build_inputs_dataframe(sim_inputs)

        with st.spinner("Optimisation économique annuelle en cours..."):
            result = optimize_dispatch_dp(sim_inputs)

        # Actual recovered curtailed PV = what the DP used from recoverable stream
        pv_curtailed_to_battery_actual = result["pv_curtailed_to_battery"]
        pv_curtailed_residual_lost_mwh = np.maximum(pv_curtailment_candidate_mwh - pv_curtailed_to_battery_actual, 0.0)

        curtailment_outputs = {
            "base_pv_generation_mwh": base_pv_hourly_mwh,
            "pv_after_tso_dso_curtailment_mwh": pv_after_tso_dso,
            "pv_after_self_curtailment_mwh": pv_after_self,
            "tso_dso_curtailed_mwh": tso_out["tso_dso_curtailed_mwh"],
            "self_curtailed_mwh": self_out["self_curtailed_mwh"],
            "pv_curtailment_candidate_mwh": pv_curtailment_candidate_mwh,
            "pv_curtailed_to_battery_mwh_actual": pv_curtailed_to_battery_actual,
            "pv_curtailed_residual_lost_mwh": pv_curtailed_residual_lost_mwh,
            "pv_effective_price_eur_per_mwh": pv_effective_price_for_revenue,
            "tso_dso_curtailment_flag": tso_out["tso_dso_curtailment_flag"],
            "self_curtailment_flag": self_out["self_curtailment_flag"],
            "self_curtailment_reason": self_out["self_curtailment_reason"],
            "pv_commercial_structure_hourly": self_out["pv_commercial_structure_hourly"],
        }

        afrr_result = None
        reconciliation = None
        final_result = result

        if sim_inputs.enable_afrr:
            with st.spinner("Simulation aFRR quart-horaire de nuit en cours..."):
                afrr_result = simulate_afrr_night_arbitrage(sim_inputs, result)
                reconciliation = reconcile_wholesale_afrr_dispatch_qh(result_hourly=result, afrr_result=afrr_result, inputs=sim_inputs)
                final_result = build_final_result_after_market_arbitration(base_result=result, reconciliation=reconciliation, inputs=sim_inputs)

        if reconciliation is not None:
            combined_qh_df = pd.DataFrame({
                "datetime": reconciliation["datetime_qh"],
                "combined_charge_to_soc_qh_mwh": reconciliation["combined_charge_to_soc_qh_mwh"],
                "combined_discharge_from_soc_qh_mwh": reconciliation["combined_discharge_from_soc_qh_mwh"],
                "wholesale_charge_to_soc_qh_mwh": (reconciliation["wholesale_pv_to_batt_qh_mwh"] + reconciliation["wholesale_grid_charge_qh_mwh"]) * sim_inputs.eta_charge,
                "wholesale_discharge_from_soc_qh_mwh": reconciliation["wholesale_discharge_qh_mwh"] / max(sim_inputs.eta_discharge, 1e-12),
                "afrr_charge_to_soc_qh_mwh": reconciliation["afrr_charge_qh_mwh"] * sim_inputs.eta_charge,
                "afrr_discharge_from_soc_qh_mwh": reconciliation["afrr_discharge_qh_mwh"] / max(sim_inputs.eta_discharge, 1e-12),
                "afrr_charge_market_qh_mwh": reconciliation["afrr_charge_qh_mwh"],
                "afrr_discharge_market_qh_mwh": reconciliation["afrr_discharge_qh_mwh"],
                "selected_discharge_channel_qh": reconciliation["selected_discharge_channel_qh"],
                "selected_discharge_price_qh": reconciliation["selected_discharge_price_qh"],
                "battery_soc_mwh_end_qh": reconciliation["combined_soc_qh"][1:],
            })
            combined_soc_hourly_end = reconciliation["combined_soc_hourly_end_mwh"]
        else:
            combined_soc_result = build_combined_soc_with_afrr(
                result_hourly=result,
                afrr_result=None,
                batt_energy_mwh=sim_inputs.batt_energy_mwh,
                initial_soc_mwh=sim_inputs.initial_soc_mwh,
                eta_charge=sim_inputs.eta_charge,
                eta_discharge=sim_inputs.eta_discharge,
            )

            combined_qh_df = pd.DataFrame({
                "datetime": build_quarter_hour_index(DEFAULT_YEAR),
                "combined_charge_to_soc_qh_mwh": combined_soc_result["combined_charge_to_soc_qh"],
                "combined_discharge_from_soc_qh_mwh": combined_soc_result["combined_discharge_from_soc_qh"],
                "wholesale_charge_to_soc_qh_mwh": combined_soc_result["wholesale_charge_to_soc_qh"],
                "wholesale_discharge_from_soc_qh_mwh": combined_soc_result["wholesale_discharge_from_soc_qh"],
                "afrr_charge_to_soc_qh_mwh": combined_soc_result["afrr_charge_to_soc_qh"],
                "afrr_discharge_from_soc_qh_mwh": combined_soc_result["afrr_discharge_from_soc_qh"],
                "afrr_charge_market_qh_mwh": combined_soc_result["afrr_charge_market_qh"],
                "afrr_discharge_market_qh_mwh": combined_soc_result["afrr_discharge_market_qh"],
                "selected_discharge_channel_qh": np.full(QH_PER_YEAR, "none", dtype=object),
                "selected_discharge_price_qh": np.full(QH_PER_YEAR, np.nan, dtype=float),
                "battery_soc_mwh_end_qh": combined_soc_result["combined_soc_qh"][1:],
            })
            combined_soc_hourly_end = combined_soc_result["combined_soc_hourly_end"]

        summary_df = build_summary_table(
            final_result,
            pv_stats,
            pure_pv_benchmark,
            pv_dc_mw,
            batt_power_mw,
            pv_capture_rate_pct,
            bess_capture_rate_pct,
            curtailment_outputs,
        )

        monthly_df = monthly_dataframe(final_result, pure_pv_benchmark, pv_dc_mw, batt_power_mw, curtailment_outputs)

        idx = pd.date_range(f"{DEFAULT_YEAR}-01-01 00:00:00", periods=HOURS_PER_YEAR, freq="h")
        hourly_df = pd.DataFrame({
            "datetime": idx,
            "base_pv_generation_mwh": base_pv_hourly_mwh,
            "pv_after_tso_dso_curtailment_mwh": pv_after_tso_dso,
            "pv_after_self_curtailment_mwh": pv_after_self,
            "pv_curtailment_candidate_mwh": pv_curtailment_candidate_mwh,
            "pv_curtailed_to_battery_mwh": pv_curtailed_to_battery_actual,
            "pv_curtailed_residual_lost_mwh": pv_curtailed_residual_lost_mwh,
            "tso_dso_curtailment_flag": tso_out["tso_dso_curtailment_flag"],
            "self_curtailment_flag": self_out["self_curtailment_flag"],
            "self_curtailment_reason": self_out["self_curtailment_reason"],
            "pv_commercial_structure": self_out["pv_commercial_structure_hourly"],
            "pv_price_raw_eur_per_mwh": pv_price_curve_raw,
            "pv_price_effective_eur_per_mwh": pv_effective_price_for_revenue,
            "pv_only_direct_mwh": pure_pv_benchmark["pv_only_direct_mwh"],
            "pv_only_revenue_eur": pure_pv_benchmark["pv_only_revenue_eur"],
            "battery_sell_price_raw_eur_per_mwh": batt_sell_curve_raw,
            "battery_sell_price_effective_eur_per_mwh": batt_sell_curve_effective,
            "grid_buy_price_raw_eur_per_mwh": grid_buy_curve_raw,
            "grid_buy_price_effective_eur_per_mwh": grid_buy_curve_effective,
            "pv_direct_mwh": final_result["pv_direct"],
            "pv_to_battery_mwh": final_result["pv_to_batt"],
            "grid_charge_mwh": final_result["grid_charge"],
            "battery_discharge_mwh": final_result["discharge"],
            "battery_soc_mwh_end": combined_soc_hourly_end,
            "pv_direct_revenue_eur": final_result["pv_direct_revenue"],
            "battery_sale_revenue_eur": final_result["batt_sale_revenue"],
            "grid_charge_cost_eur": final_result["grid_charge_cost"],
            "avg_stored_charge_price_eur_per_mwh": final_result["avg_stored_charge_price"][1:],
            "required_discharge_price_eur_per_mwh": final_result["required_discharge_price"],
            "required_discharge_price_gate_estimate_eur_per_mwh": final_result["required_discharge_price_gate_estimate"],
            "afrr_charge_mwh": final_result["afrr_charge_hourly_mwh"] if "afrr_charge_hourly_mwh" in final_result else np.zeros(HOURS_PER_YEAR),
            "afrr_discharge_mwh": final_result["afrr_discharge_hourly_mwh"] if "afrr_discharge_hourly_mwh" in final_result else np.zeros(HOURS_PER_YEAR),
            "afrr_charge_cost_eur": final_result["afrr_charge_cost_hourly_eur"] if "afrr_charge_cost_hourly_eur" in final_result else np.zeros(HOURS_PER_YEAR),
            "afrr_sale_revenue_eur": final_result["afrr_sale_revenue_hourly_eur"] if "afrr_sale_revenue_hourly_eur" in final_result else np.zeros(HOURS_PER_YEAR),
            "afrr_cycle_cost_eur": final_result["afrr_cycle_cost_hourly_eur"] if "afrr_cycle_cost_hourly_eur" in final_result else np.zeros(HOURS_PER_YEAR),
            "afrr_net_revenue_eur": final_result["afrr_net_revenue_hourly_eur"] if "afrr_net_revenue_hourly_eur" in final_result else np.zeros(HOURS_PER_YEAR),
            "pv_capture_rate_pct": np.full(HOURS_PER_YEAR, pv_capture_rate_pct),
            "bess_capture_rate_pct": np.full(HOURS_PER_YEAR, bess_capture_rate_pct),
        })

        afrr_qh_df = None
        if reconciliation is not None:
            afrr_qh_df = pd.DataFrame({
                "datetime": reconciliation["datetime_qh"],
                "afrr_charge_price_raw_eur_per_mwh": afrr_charge_curve_qh_raw if afrr_charge_curve_qh_raw is not None else np.zeros(QH_PER_YEAR),
                "afrr_charge_price_effective_eur_per_mwh": sim_inputs.afrr_charge_price_qh,
                "afrr_discharge_price_raw_eur_per_mwh": afrr_discharge_curve_qh_raw if afrr_discharge_curve_qh_raw is not None else np.zeros(QH_PER_YEAR),
                "afrr_discharge_price_effective_eur_per_mwh": sim_inputs.afrr_discharge_price_qh,
                "afrr_charge_mwh": reconciliation["afrr_charge_qh_mwh"],
                "afrr_discharge_mwh": reconciliation["afrr_discharge_qh_mwh"],
                "wholesale_discharge_mwh": reconciliation["wholesale_discharge_qh_mwh"],
                "selected_discharge_channel": reconciliation["selected_discharge_channel_qh"],
                "selected_discharge_price_eur_per_mwh": reconciliation["selected_discharge_price_qh"],
                "combined_soc_mwh": reconciliation["combined_soc_qh"][1:],
                "afrr_charge_cost_eur": reconciliation["afrr_charge_cost_qh_eur"],
                "afrr_sale_revenue_eur": reconciliation["afrr_sale_revenue_qh_eur"],
                "afrr_cycle_cost_eur": reconciliation["afrr_cycle_cost_qh_eur"],
                "afrr_net_revenue_eur": reconciliation["afrr_net_revenue_qh_eur"],
                "bess_capture_rate_pct": np.full(QH_PER_YEAR, bess_capture_rate_pct),
            })

        inputs_df = build_inputs_dataframe(sim_inputs)
        excel_bytes = to_excel_bytes(
            inputs_df=inputs_df,
            summary_df=summary_df,
            monthly_df=monthly_df,
            hourly_df=hourly_df,
            afrr_qh_df=afrr_qh_df,
            afrr_daily_log_df=afrr_result["afrr_daily_log"] if afrr_result is not None else None,
            bess_degradation_df=bess_degradation_df,
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time < 60:
            optimization_time_str = f"{elapsed_time:.2f} seconds"
        else:
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            optimization_time_str = f"{minutes} min {seconds} sec"

        st.subheader("Optimization Time")
        st.write(optimization_time_str)

        st.success("Simulation terminée.")

        k1, k2, k3, k4 = st.columns(4)
        total_revenue_display = final_result["total_revenue_including_afrr_eur"][0] if "total_revenue_including_afrr_eur" in final_result else final_result["total_revenue"][0]
        total_energy_display = final_result["energy_sold_total_mwh"][0] + (np.sum(final_result["afrr_discharge_hourly_mwh"]) if "afrr_discharge_hourly_mwh" in final_result else 0.0)

        k1.metric("Revenu total", f"{total_revenue_display:,.0f} EUR")
        k2.metric("Énergie totale vendue", f"{total_energy_display:,.0f} MWh")
        k3.metric("Énergie shiftée", f"{final_result['energy_shifted_mwh'][0]:,.0f} MWh")
        k4.metric("Cycles équivalents", f"{final_result['equivalent_cycles'][0]:,.1f}")

        st.subheader("Synthèse")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        debug = hourly_df[
            (hourly_df["datetime"] >= pd.Timestamp(f"{DEFAULT_YEAR}-06-01 00:00:00")) &
            (hourly_df["datetime"] < pd.Timestamp(f"{DEFAULT_YEAR}-06-04 00:00:00"))
        ].copy()

        st.subheader("Debug curtailment (3 premiers jours de juin)")
        st.dataframe(
            debug[[
                "datetime",
                "base_pv_generation_mwh",
                "pv_after_tso_dso_curtailment_mwh",
                "pv_after_self_curtailment_mwh",
                "pv_curtailment_candidate_mwh",
                "pv_curtailed_to_battery_mwh",
                "pv_curtailed_residual_lost_mwh",
                "pv_price_raw_eur_per_mwh",
                "pv_price_effective_eur_per_mwh",
                "self_curtailment_flag",
                "self_curtailment_reason",
                "pv_commercial_structure",
            ]],
            use_container_width=True,
        )

        c1, c2 = st.columns(2)

        with c1:
            fig1, ax1 = plt.subplots(figsize=(8, 4.5))
            bars = [
                float(final_result["total_direct_pv_revenue"][0]),
                float(final_result["total_batt_sale_revenue"][0]),
                -float(final_result["total_grid_charge_cost"][0]),
                float(final_result["nightly_revenue_total"][0]),
                float(final_result["total_afrr_net_revenue_eur"][0]) if "total_afrr_net_revenue_eur" in final_result else 0.0,
                float(pure_pv_benchmark["total_pv_only_revenue_eur"][0]),
            ]
            labels = ["PV direct", "Vente batterie", "Coût charge réseau", "SS nuit", "aFRR net", "PV-only"]
            ax1.bar(labels, bars)
            ax1.set_title("Décomposition des revenus")
            ax1.set_ylabel("EUR")
            ax1.tick_params(axis="x", rotation=20)
            st.pyplot(fig1)
            plt.close(fig1)

        with c2:
            fig2, ax2 = plt.subplots(figsize=(9, 4.8))

            x = np.arange(len(monthly_df))
            pv_vals = monthly_df["pv_revenue_keur_per_mw"].to_numpy(dtype=float)
            afrr_vals = monthly_df["afrr_net_revenue"].to_numpy(dtype=float) / max(batt_power_mw, 1e-12) / 1000.0
            bess_vals = monthly_df["bess_revenue_keur_per_mw"].to_numpy(dtype=float) - afrr_vals

            ax2.bar(x, bess_vals, width=0.65, color="green", label="BESS")
            ax2.bar(x, afrr_vals, width=0.65, bottom=bess_vals, color="blue", label="aFRR")
            ax2.bar(x, pv_vals, width=0.65, bottom=bess_vals + afrr_vals, color="orange", label="PV")

            ax2.set_title("Revenus mensuels spécifiques superposés")
            ax2.set_ylabel("kEUR/MW")
            ax2.set_xlabel("Mois")
            ax2.set_xticks(x)
            ax2.set_xticklabels(monthly_df["month"], rotation=45)
            ax2.legend()

            st.pyplot(fig2)
            plt.close(fig2)

        c3, c4 = st.columns(2)

        with c3:
            fig3, ax3 = plt.subplots(figsize=(8, 4.5))

            ax3.plot(monthly_df["month"], monthly_df["pv_direct_mwh"], label="PV direct")
            ax3.plot(monthly_df["month"], monthly_df["shifted_mwh"], label="Énergie shiftée wholesale")
            ax3.plot(monthly_df["month"], monthly_df["pv_only_direct_mwh"], label="PV-only direct")

            if "afrr_discharge_mwh" in monthly_df.columns:
                ax3.plot(monthly_df["month"], monthly_df["afrr_discharge_mwh"], label="Décharge aFRR")

            if "pv_curtailment_candidate_mwh" in monthly_df.columns:
                ax3.plot(
                    monthly_df["month"],
                    monthly_df["pv_curtailment_candidate_mwh"],
                    linestyle="--",
                    marker="o",
                    label="PV curtailed"
                )

            if "pv_curtailed_to_battery_mwh_actual" in monthly_df.columns:
                ax3.plot(
                    monthly_df["month"],
                    monthly_df["pv_curtailed_to_battery_mwh_actual"],
                    linestyle="--",
                    marker="o",
                    label="PV curtailed → battery"
                )

            if "pv_curtailed_residual_lost_mwh" in monthly_df.columns:
                ax3.plot(
                    monthly_df["month"],
                    monthly_df["pv_curtailed_residual_lost_mwh"],
                    linestyle="--",
                    marker="o",
                    label="PV curtailed lost"
                )

            ax3.set_title("Énergies valorisées par mois")
            ax3.set_ylabel("MWh")
            ax3.set_xlabel("Mois")
            ax3.legend()
            ax3.tick_params(axis="x", rotation=45)
            st.pyplot(fig3)
            plt.close(fig3)

        with c4:
            start_date = pd.Timestamp(f"{DEFAULT_YEAR}-06-01 00:00:00")
            end_date = start_date + pd.Timedelta(hours=120)

            df_plot = hourly_df[
                (hourly_df["datetime"] >= start_date) &
                (hourly_df["datetime"] < end_date)
            ].copy()

            fig, ax1 = plt.subplots(figsize=(12, 5))
            bar_width = 0.03

            ax1.fill_between(
                df_plot["datetime"],
                df_plot["pv_direct_mwh"],
                color="orange",
                alpha=0.5,
                label="PV → Réseau"
            )
            ax1.plot(
                df_plot["datetime"],
                df_plot["pv_direct_mwh"],
                color="orange",
                linewidth=1.8
            )

            ax1.bar(
                df_plot["datetime"],
                df_plot["battery_discharge_mwh"],
                width=bar_width,
                label="Batterie → Réseau (wholesale)",
                alpha=0.8,
                color="green"
            )

            ax1.bar(
                df_plot["datetime"],
                -df_plot["pv_to_battery_mwh"],
                width=bar_width,
                label="PV → Batterie",
                alpha=0.6,
                color="red"
            )

            ax1.bar(
                df_plot["datetime"],
                -df_plot["grid_charge_mwh"],
                width=bar_width,
                bottom=-df_plot["pv_to_battery_mwh"],
                label="Réseau → Batterie",
                alpha=0.6
            )

            if "afrr_discharge_mwh" in df_plot.columns:
                ax1.bar(
                    df_plot["datetime"],
                    df_plot["afrr_discharge_mwh"],
                    width=bar_width,
                    label="aFRR → Décharge",
                    alpha=0.5,
                    color="purple"
                )

            if "afrr_charge_mwh" in df_plot.columns:
                ax1.bar(
                    df_plot["datetime"],
                    -df_plot["afrr_charge_mwh"],
                    width=bar_width,
                    label="aFRR → Charge",
                    alpha=0.5,
                    color="blue"
                )

            if "pv_curtailment_candidate_mwh" in df_plot.columns:
                ax1.plot(
                    df_plot["datetime"],
                    df_plot["pv_curtailment_candidate_mwh"],
                    linestyle="--",
                    linewidth=1.5,
                    label="PV curtailed"
                )

            if "pv_curtailed_to_battery_mwh" in df_plot.columns:
                ax1.bar(
                    df_plot["datetime"],
                    -df_plot["pv_curtailed_to_battery_mwh"],
                    width=bar_width,
                    label="PV curtailed → battery",
                    alpha=0.6
                )

            if "pv_curtailed_residual_lost_mwh" in df_plot.columns:
                ax1.plot(
                    df_plot["datetime"],
                    df_plot["pv_curtailed_residual_lost_mwh"],
                    linestyle=":",
                    linewidth=1.8,
                    label="PV curtailed lost"
                )

            ax1.axhline(0, linewidth=1)
            ax1.set_ylabel("Flux énergie (MWh)")
            ax1.set_xlabel("Heure")
            ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Hh"))
            ax1.tick_params(axis="x", rotation=0)

            ax2 = ax1.twinx()
            ax2.plot(
                df_plot["datetime"],
                df_plot["pv_price_effective_eur_per_mwh"],
                linestyle="--",
                alpha=0.7,
                label="Prix spot PV effectif"
            )
            ax2.set_ylabel("Prix (EUR/MWh)")

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(
                lines_1 + lines_2,
                labels_1 + labels_2,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=3,
                frameon=False,
            )
            ax1.set_title("Dispatch énergétique - 5 premiers jours de juin")
            fig.tight_layout(rect=[0, 0.12, 1, 1])

            st.pyplot(fig)
            plt.close(fig)

        c5, c6 = st.columns(2)

        with c5:
            fig5, ax5 = plt.subplots(figsize=(9, 4.8))

            x = np.arange(len(monthly_df))
            width = 0.34

            pv_vals_mwh = monthly_df["pv_revenue_eur_per_mwh"].to_numpy(dtype=float)
            bess_vals_mwh = monthly_df["bess_revenue_eur_per_mwh"].to_numpy(dtype=float)
            pv_only_vals_mwh = (
                monthly_df["pv_only_revenue"].to_numpy(dtype=float)
                / monthly_df["pv_only_direct_mwh"].clip(lower=1e-12).to_numpy(dtype=float)
            )

            ax5.bar(
                x - width / 2,
                pv_vals_mwh,
                width=width,
                color="orange",
                label="PV hybride"
            )

            ax5.bar(
                x + width / 2,
                bess_vals_mwh,
                width=width,
                color="green",
                label="BESS"
            )

            ax5.plot(
                x,
                pv_only_vals_mwh,
                marker="o",
                linewidth=2.0,
                label="PV-only Project"
            )

            ax5.set_title("Revenus mensuels spécifiques énergie")
            ax5.set_ylabel("EUR/MWh")
            ax5.set_xlabel("Mois")
            ax5.set_xticks(x)
            ax5.set_xticklabels(monthly_df["month"], rotation=45)
            ax5.legend()

            st.pyplot(fig5)
            plt.close(fig5)

        with c6:
            if afrr_qh_df is not None:
                qh_debug_start = pd.Timestamp(f"{DEFAULT_YEAR}-06-01 00:00:00")
                qh_debug_end = qh_debug_start + pd.Timedelta(days=3)

                qh_plot = afrr_qh_df[
                    (afrr_qh_df["datetime"] >= qh_debug_start) &
                    (afrr_qh_df["datetime"] < qh_debug_end)
                ].copy()

                fig6, ax6 = plt.subplots(figsize=(12, 4.8))
                ax6.bar(qh_plot["datetime"], qh_plot["afrr_discharge_mwh"], width=0.008, label="Décharge aFRR", alpha=0.7)
                ax6.bar(qh_plot["datetime"], qh_plot["wholesale_discharge_mwh"], width=0.008, label="Décharge wholesale", alpha=0.7)
                ax6.bar(qh_plot["datetime"], -qh_plot["afrr_charge_mwh"], width=0.008, label="Charge aFRR", alpha=0.7)
                ax6.set_ylabel("MWh / 15 min")
                ax6.set_title("Arbitrage quart-horaire - 3 premiers jours de juin")
                ax6.xaxis.set_major_locator(mdates.HourLocator(interval=6))
                ax6.xaxis.set_major_formatter(mdates.DateFormatter("%d %Hh"))
                ax6.tick_params(axis="x", rotation=45)

                ax6b = ax6.twinx()
                ax6b.plot(qh_plot["datetime"], qh_plot["afrr_charge_price_effective_eur_per_mwh"], linestyle="--", alpha=0.7, label="Prix charge aFRR effectif")
                ax6b.plot(qh_plot["datetime"], qh_plot["afrr_discharge_price_effective_eur_per_mwh"], linestyle="-.", alpha=0.7, label="Prix décharge aFRR effectif")
                ax6b.set_ylabel("EUR/MWh")

                lines_a, labels_a = ax6.get_legend_handles_labels()
                lines_b, labels_b = ax6b.get_legend_handles_labels()
                ax6.legend(lines_a + lines_b, labels_a + labels_b, loc="upper right")

                st.pyplot(fig6)
                plt.close(fig6)
            else:
                st.info("Activez l'aFRR et uploadez les deux fichiers quart-horaires pour afficher le graphique aFRR.")

        c7, c8 = st.columns(2)

        with c7:
            st.subheader("Comparaison Revenu PV-only vs Hybrid")

            fig_cmp, ax_cmp = plt.subplots(figsize=(9, 4.8))

            x = np.arange(len(monthly_df))

            pv_only_monthly_keur = monthly_df["pv_only_revenue"].to_numpy(dtype=float) / 1000.0
            hybrid_monthly_keur = monthly_df["net_revenue"].to_numpy(dtype=float) / 1000.0

            ax_cmp.plot(
                x,
                pv_only_monthly_keur,
                marker="o",
                linewidth=2.0,
                label="PV-only"
            )

            ax_cmp.plot(
                x,
                hybrid_monthly_keur,
                marker="o",
                linewidth=2.0,
                label="Hybrid (PV + BESS)"
            )

            ax_cmp.set_title("Comparaison Revenu PV-only vs Hybrid")
            ax_cmp.set_ylabel("kEUR")
            ax_cmp.set_xlabel("Mois")
            ax_cmp.set_xticks(x)
            ax_cmp.set_xticklabels(monthly_df["month"], rotation=45)
            ax_cmp.legend()

            st.pyplot(fig_cmp)
            plt.close(fig_cmp)

        with c8:
            fig8, ax8 = plt.subplots(figsize=(9, 4.8))

            x = np.arange(len(monthly_df))
            width = 0.26

            if "pv_curtailment_candidate_mwh" in monthly_df.columns:
                ax8.bar(
                    x - width,
                    monthly_df["pv_curtailment_candidate_mwh"].to_numpy(dtype=float),
                    width=width,
                    label="PV curtailed"
                )

            if "pv_curtailed_to_battery_mwh_actual" in monthly_df.columns:
                ax8.bar(
                    x,
                    monthly_df["pv_curtailed_to_battery_mwh_actual"].to_numpy(dtype=float),
                    width=width,
                    label="PV curtailed → battery"
                )

            if "pv_curtailed_residual_lost_mwh" in monthly_df.columns:
                ax8.bar(
                    x + width,
                    monthly_df["pv_curtailed_residual_lost_mwh"].to_numpy(dtype=float),
                    width=width,
                    label="PV curtailed lost"
                )

            ax8.set_title("Curtailment mensuel PV")
            ax8.set_ylabel("MWh")
            ax8.set_xlabel("Mois")
            ax8.set_xticks(x)
            ax8.set_xticklabels(monthly_df["month"], rotation=45)
            ax8.legend()

            st.pyplot(fig8)
            plt.close(fig8)

        st.subheader("Table mensuelle")
        st.dataframe(monthly_df, use_container_width=True, hide_index=True)

        st.subheader("Exports")
        st.download_button(
            "Télécharger cette simulation complète (Excel)",
            data=excel_bytes,
            file_name="simulation_complete_hybride_pv_bess.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "Télécharger l'horaire en CSV",
            data=hourly_df.to_csv(index=False).encode("utf-8"),
            file_name="dispatch_horaire_hybride.csv",
            mime="text/csv",
        )

        if afrr_qh_df is not None:
            st.download_button(
                "Télécharger l'aFRR quart-horaire en CSV",
                data=afrr_qh_df.to_csv(index=False).encode("utf-8"),
                file_name="dispatch_afrr_quart_horaire.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Erreur: {e}")


if __name__ == "__main__":
    app()
