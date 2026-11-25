"""Persistent surface coordinates and observer-local live queries."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from o2xygen.atmosphere import default_atmosphere
from o2xygen.constants import AU, DAY, L_SUN, M_EARTH, M_SUN, R_SUN
from o2xygen.observation import normalize_point, local_frame, enu_angles
from o2xygen.system import System, Body, STAR, BODY
from o2xygen.world import system_to_dict, dict_to_system, load_world
from workspace import Workspace


def test_v3_point_is_only_name_body_lat_lon_and_legacy_loads():
    s = System("surface")
    s.add(Body("S", STAR, mass_kg=M_SUN))
    s.add(Body("P", BODY, mass_kg=M_EARTH, radius_m=6.371e6))
    s.observation_points = [{"name":"zero", "body":"P",
                            "latitude_deg":0, "longitude_deg":0}]
    raw = system_to_dict(s)
    assert raw["version"] == 3
    assert raw["observations"] == [{"name":"zero", "body":"P",
                                   "latitude_deg":0.0, "longitude_deg":0.0}]
    assert set(raw["observations"][0]) == {"name","body","latitude_deg","longitude_deg"}
    raw.pop("observations"); raw["version"] = 2
    assert dict_to_system(raw).observation_points == []


def test_longitude_canonicalization_and_invalid_star_point():
    assert normalize_point({"name":"x","body":"p","latitude_deg":0,
                            "longitude_deg":360})["longitude_deg"] == 0.0
    assert normalize_point({"name":"x","body":"p","latitude_deg":0,
                            "longitude_deg":180})["longitude_deg"] == -180.0
    s = System("bad")
    s.add(Body("S", STAR, mass_kg=M_SUN, radius_m=R_SUN))
    s.observation_points = [{"name":"bad","body":"S","latitude_deg":0,"longitude_deg":0}]
    with pytest.raises(ValueError, match="only on non-stellar"):
        system_to_dict(s)


def test_equatorial_basis_and_phase_change_surface_direction():
    body = {"p":np.array([10.0,20.0,30.0]),"radius_m":2.0}
    p = {"name":"E0","body":"P","latitude_deg":0.0,"longitude_deg":0.0}
    f0 = local_frame(body,{"axis":[0,0,1],"phase_rad":0.0},p)
    np.testing.assert_allclose(f0["up"],[1,0,0],atol=1e-15)
    np.testing.assert_allclose(f0["east"],[0,1,0],atol=1e-15)
    np.testing.assert_allclose(f0["north"],[0,0,1],atol=1e-15)
    f1 = local_frame(body,{"axis":[0,0,1],"phase_rad":np.pi/2},p)
    np.testing.assert_allclose(f1["up"],[0,1,0],atol=1e-15)
    for f in (f0,f1):
        basis=np.array([f["east"],f["north"],f["up"]])
        np.testing.assert_allclose(basis@basis.T,np.eye(3),atol=1e-14)
    assert enu_angles([1,0,0])["azimuth_deg"] == 90.0
    assert enu_angles([0,0,-1])["altitude_deg"] == -90.0


def _observer_world():
    s=System("observer")
    # A star in the local zenith direction of a northern observation point.
    s.add(Body("near",STAR,mass_kg=M_SUN,pos0=np.array([150e6,0.,100e6]),
               vel0=np.array([0.,1000.,0.]),radius_m=1e6,luminosity_w=L_SUN))
    s.add(Body("far",STAR,mass_kg=M_SUN,pos0=np.array([250e6,0.,100e6]),
               vel0=np.array([0.,1000.,0.]),radius_m=1e6,luminosity_w=L_SUN))
    s.add(Body("P",BODY,mass_kg=M_EARTH,pos0=np.array([0.,0.,100e6]),
               vel0=np.array([0.,1000.,0.]),radius_m=10e6,spin_axis=np.array([0.,0.,1.])))
    s.bodies["P"].extra["atmosphere"]=default_atmosphere()
    s.observation_points=[{"name":"site","body":"P","latitude_deg":0.0,"longitude_deg":0.0}]
    return s


def test_observer_api_uses_current_phase_ground_and_finite_occultation(monkeypatch):
    import app
    s=_observer_world();ws=Workspace(s,grid_getter=None)
    monkeypatch.setattr(app,"SYSTEM",s);monkeypatch.setattr(app,"WS",ws)
    monkeypatch.setattr(app,"_sky_payload",lambda *args:{"hip":[],"dir_x":[],"dir_y":[],
        "dir_z":[],"mag":[],"bv":[],"sun_index":None})
    result=app.api_get_observation({"name":["site"]})
    assert result["t_days"] == 0.0
    by_name={b["name"]:b for b in result["bodies"]}
    assert by_name["near"]["above_horizon"] is True
    assert by_name["near"]["altitude_deg"] == pytest.approx(90.0)
    assert by_name["near"]["occluded_by"] == []
    assert by_name["far"]["occluded_by"] == ["near"]
    assert result["illumination"]["total"] > 0.0
    assert result["surface_irradiance_wm2"]["far"] == 0.0
    assert result["surface_irradiance_wm2"]["near"] > 0.0
    # Equator 0°E turns to the night side after half a rotation, then back.
    s.bodies["P"].spin_phase_rad=np.pi
    night=app.api_get_observation({"name":["site"]})
    assert night["bodies"][0]["above_horizon"] is False
    assert night["surface_irradiance_wm2"]["total"] == 0.0
    s.bodies["P"].spin_phase_rad=0.0
    day=app.api_get_observation({"name":["site"]})
    assert day["bodies"][0]["above_horizon"] is True
    assert day["surface_irradiance_wm2"]["total"] > 0.0
    assert len(day["position_m"]) == 3


def test_phase_advances_and_survives_save_reload(tmp_path):
    s=System("spin")
    s.add(Body("P",BODY,mass_kg=M_EARTH,radius_m=6.371e6,
               spin_rate_rad_s=2*np.pi/DAY,spin_phase_rad=0.3))
    s.observation_points=[{"name":"E0","body":"P","latitude_deg":0.0,"longitude_deg":0.0}]
    ws=Workspace(s,grid_getter=None)
    expected=(0.3+np.pi)%(2*np.pi)
    ws.advance_to(0.5)
    assert ws.state_at(None)["spins"]["P"]["phase_rad"] == pytest.approx(expected)
    out=ws.snapshot_for_save()
    assert out.bodies["P"].spin_phase_rad == pytest.approx(expected)
    assert out.observation_points == s.observation_points
    from o2xygen.world import save_world
    back=load_world(save_world(out,tmp_path/"spin"))
    assert back.bodies["P"].spin_phase_rad == pytest.approx(expected)


def test_observation_crud_confirm_and_nonstar_spin_edit(tmp_path,monkeypatch):
    import app
    s=_observer_world();ws=Workspace(s,grid_getter=None)
    monkeypatch.setattr(app,"SYSTEM",s);monkeypatch.setattr(app,"WS",ws)
    monkeypatch.setattr(app,"CURRENT_WORLD","observer")
    monkeypatch.setattr(app,"WORLDS_DIR",tmp_path)
    monkeypatch.setattr(app,"_evolution_active",lambda:False)
    with pytest.raises(app.HttpError) as e:
        app.api_save_observation({"name":"E0","body":"P","latitude_deg":0,"longitude_deg":0})
    assert e.value.status == 400
    assert not app._world_path("observer").exists()
    got=app.api_save_observation({"name":"E0","body":"P","latitude_deg":0,
                                  "longitude_deg":0,"confirm":True})
    assert got["point"]["longitude_deg"] == 0.0
    assert load_world(app._world_path("observer")).observation_points[-1]["name"] == "E0"
    with pytest.raises(app.HttpError):
        app.api_save_observation({"name":"E0","body":"P","latitude_deg":0,
                                  "longitude_deg":1,"confirm":True})
    app.api_save_observation({"original_name":"E0","name":"renamed","body":"P",
                              "latitude_deg":10,"longitude_deg":20,"confirm":True})
    assert app.api_list_observations()["points"][-1]["name"] == "renamed"
    old_positions=ws.sim.live_state()
    app.api_body_edit({"name":"P","spin_period_h":10.0,"spin_phase_deg":30.0,
                       "spin_axis":[0,0,1],"atmosphere":default_atmosphere()})
    assert s.bodies["P"].rotation_period_s == pytest.approx(36000.0)
    assert s.bodies["P"].spin_phase_rad == pytest.approx(np.pi/6)
    for n,(p,v) in ws.sim.live_state().items():
        np.testing.assert_array_equal(p,old_positions[n][0])
        np.testing.assert_array_equal(v,old_positions[n][1])
    app.api_body_edit({"name":"P","atmosphere":None})
    assert "atmosphere" not in s.bodies["P"].extra
    with pytest.raises(app.HttpError) as e:
        app.api_delete_observation({"name":"renamed"})
    assert e.value.status == 400
    app.api_delete_observation({"name":"renamed","confirm":True})
    assert [p["name"] for p in app.api_list_observations()["points"]] == ["site"]
