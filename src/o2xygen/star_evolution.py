"""Star evolution wrapper: the logic behind his 参数估算结果 modal.

Bookkeeping rules transcribed from the modal text:
  累计损失 = 初始质量 - 轨道当前质量
  损失率来自轨道 star_mdot，不等于累计损失除以年龄
  应用后更新半径/光度/有效温度；当前质量仅在某选项后更新
  (初始质量保持不变；轨道按更新后的质量重算)
"""
from dataclasses import dataclass
from .stars import MistGrid, StarProperties


@dataclass
class StarEstimate:
    initial_mass_msun: float
    age_yr: float
    feh: float
    props: StarProperties

    @property
    def cumulative_loss_msun(self) -> float:
        """初始质量 - 当前质量 (his definition, not mdot * age)."""
        return self.initial_mass_msun - self.props.mass_msun

    @property
    def loss_rate_msun_per_yr(self) -> float:
        """当前损失率, straight from the track's star_mdot."""
        return self.props.mdot_msun_per_yr


class StarEstimator:
    """Drives a MistGrid the way his 一键计算 star stage does."""

    def __init__(self, grid: MistGrid):
        self.grid = grid

    def estimate(self, initial_mass_msun: float, feh: float, age_yr: float) -> StarEstimate:
        props = self.grid.evaluate(initial_mass_msun, feh, age_yr)
        return StarEstimate(initial_mass_msun, age_yr, feh, props)

    def apply_to_body(self, system, star_name: str, est: StarEstimate,
                      update_mass: bool = False) -> None:
        """应用估算结果: update luminosity/radius/teff (+mass if opted in).
        Orbit recomputation with new mass is the caller's job (System uses
        parent gm at propagation time, so it is automatic)."""
        body = system.bodies[star_name]
        body.extra["teff_k"] = est.props.teff_k
        body.extra["evolved_radius_m"] = est.props.radius_m
        if update_mass:
            from .constants import M_SUN
            body.mass_kg = est.props.mass_msun * M_SUN
            # pin the MIST seed alongside the evolved mass: save_world falls
            # back to mass_kg/M_SUN when extra["mini_msun"] is absent, which
            # would otherwise persist the wind-evolved mass as the initial
            # mass across a save/load cycle
            body.extra["mini_msun"] = est.initial_mass_msun
        body.luminosity_w = est.props.luminosity_w
