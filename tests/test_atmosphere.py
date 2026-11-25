"""Spherical optical paths, real multi-order scattering and cache invalidation."""
import numpy as np
import pytest

from o2xygen.atmosphere import (SphericalAtmosphere, default_atmosphere,
    normalize_atmosphere, make_optical_table, sample_optical_table, multiple_scattering)
from o2xygen.atmosphere_cache import (QUALITY, optics_key, build_atmosphere_lut,
    load_or_build_atmosphere, spectrum_rgb)


def test_vertical_exponential_column_matches_quadrature():
    m=SphericalAtmosphere(6.371e6,default_atmosphere())
    exact=m.vertical_optical_depth()
    p=np.array([0.,0.,m.radius_m])
    tau=m.optical_depth(p,np.array([0.,0.,1.]))
    np.testing.assert_allclose(tau,exact,rtol=2e-6,atol=0)
    np.testing.assert_array_less(0.0,m.transmittance(p,[0.,0.,1.]))
    np.testing.assert_array_less(m.transmittance(p,[0.,0.,1.]),1.0)


def test_optical_table_tracks_direct_integration():
    a=default_atmosphere();m=SphericalAtmosphere(6.371e6,a)
    table=make_optical_table(m,mu_count=129,height_count=48)
    for h,mu in [(0.,1.),(1000.,0.7),(30e3,0.1),(90e3,-0.4)]:
        p=np.array([0.,0.,m.radius_m+h]);d=np.array([np.sqrt(1-mu*mu),0.,mu])
        actual=sample_optical_table(m,table,p,d)
        expected=m.optical_depth(p,d,ground=False)
        np.testing.assert_allclose(actual,expected,rtol=0.025,atol=2e-5)


def test_ground_segment_cdf_is_nonnegative_and_quadrature_converges():
    rng=np.random.default_rng(90210)
    a=default_atmosphere()
    m=SphericalAtmosphere(6.371e6,a,integration_points=48)
    reference=SphericalAtmosphere(6.371e6,a,integration_points=96)
    count=192
    p=np.tile([0.,0.,m.radius_m+3000.0],(count,1))
    mu=rng.uniform(-0.95,0.05,count)
    d=np.stack((np.sqrt(1-mu*mu),np.zeros(count),mu),axis=1)
    length=m.ray_length(p,d,ground=True)
    tau=m.optical_depth(p,d,length)
    exact=reference.optical_depth(p,d,length)
    assert np.min(tau)>=0.0
    np.testing.assert_allclose(tau,exact,rtol=0.01,atol=2e-7)


def test_grazing_solar_table_does_not_use_linear_mu_interpolation():
    m=SphericalAtmosphere(6.371e6,default_atmosphere())
    table=make_optical_table(m,mu_count=129,height_count=48)
    for h in (0.0,3000.0,30000.0):
        for mu in (0.001,0.01,0.1):
            p=np.array([0.,0.,m.radius_m+h])
            d=np.array([np.sqrt(1-mu*mu),0.,mu])
            np.testing.assert_allclose(sample_optical_table(m,table,p,d),
                m.optical_depth(p,d,ground=False),rtol=0.035,atol=2e-5)


def test_phase_functions_are_normalized_and_finite():
    m=SphericalAtmosphere(6.371e6,default_atmosphere())
    x,w=np.polynomial.legendre.leggauss(160)
    pr,pm=m.phase(x)
    assert np.dot(pr,w)*2*np.pi == pytest.approx(1.0,abs=1e-12)
    assert np.dot(pm,w)*2*np.pi == pytest.approx(1.0,abs=1e-8)
    assert np.all(np.isfinite(pm))


def test_solid_planet_blocks_sun_even_in_optically_thin_shell():
    a=default_atmosphere()
    a["rayleigh_scattering_m_inv"]=[1e-12]*3
    a["aerosol_scattering_m_inv"]=[0.0]*3
    a["absorption_m_inv"]=[0.0]*3
    m=SphericalAtmosphere(6.371e6,a)
    p=np.array([0.,0.,m.radius_m+0.01])
    np.testing.assert_array_equal(m.sun_transmittance(p,[0.,0.,-1.]),[0.,0.,0.])
    np.testing.assert_array_less(0.99,m.sun_transmittance(p,[0.,0.,1.]))


def _uniform_model():
    a=default_atmosphere()
    a.update({"height_m":1000.0,"scale_height_m":1e10,
              "rayleigh_scattering_m_inv":[1e-5]*3,
              "aerosol_scattering_m_inv":[0.0]*3,"absorption_m_inv":[0.0]*3,
              "ground_albedo":[0.0]*3})
    return SphericalAtmosphere(6.371e6,a)


def test_mc_first_order_converges_to_independent_quadrature():
    m=_uniform_model()
    r=multiple_scattering(m,[0.,0.,1.],[0.,0.,1.],samples=1024,
                          orders=1,replace_first=False)
    sampled=r["volume_orders"][0,0]
    analytic=m.single_scattering([0.,0.,m.radius_m+0.01],[0.,0.,1.],[0.,0.,1.])
    # Average the independent RGB paths: this catches a missing/extra T (50%
    # bias) without mistaking a sparse-channel Poisson fluctuation for bias.
    np.testing.assert_allclose(np.mean(sampled),np.mean(analytic),rtol=0.15,atol=1e-9)
    # Uniform thin shell: integral ~= beta*height*Rayleigh(1).
    expected=1e-5*1000.0*3/(8*np.pi)
    np.testing.assert_allclose(analytic,expected,rtol=0.01)


def test_ground_direct_limit_and_separate_scattering_orders():
    a=default_atmosphere()
    a.update({"rayleigh_scattering_m_inv":[1e-30]*3,
              "aerosol_scattering_m_inv":[0.0]*3,"absorption_m_inv":[0.0]*3,
              "ground_albedo":[0.2]*3,"aerosol_anisotropy":0.0})
    m=SphericalAtmosphere(6.371e6,a)
    r=multiple_scattering(m,[0.,0.,-1.],[0.,0.,1.],samples=64,orders=3)
    np.testing.assert_allclose(r["ground"][0],0.2/np.pi,rtol=1e-6)
    assert r["ground_orders"].shape == (3,1,3)
    assert r["volume_orders"].shape == (3,1,3)
    assert np.all(np.isfinite(r["multiple"]))


def test_cache_is_versioned_rebuildable_and_input_bound(tmp_path):
    a=default_atmosphere()
    q=dict(QUALITY,sun_count=3,view_count=3,azimuth_count=2,
           samples_per_channel=16,orders=3,mu_count=33,height_count=12)
    model=build_atmosphere_lut(6.371e6,a,q)
    assert model["multiple"].shape == (3,3,3)
    assert model["tau"].shape == (12,33,3)
    assert np.all(np.isfinite(model["multiple"]))
    assert np.min(model["multiple"]) >= 0.0
    assert np.max(model["multiple"]) > 0.0
    assert model["fingerprint"] == optics_key(6.371e6,a,q)
    assert optics_key(6.371e6,a) != optics_key(6.372e6,a)
    # Build through the actual cache interface with the tiny test quality.
    original=QUALITY.copy()
    try:
        QUALITY.update(q)
        first=load_or_build_atmosphere(tmp_path,6.371e6,a)
        second=load_or_build_atmosphere(tmp_path,6.371e6,a)
        assert first["cached"] is False and second["cached"] is True
        np.testing.assert_array_equal(first["multiple"],second["multiple"])
        assert len(list(tmp_path.glob("*.npz"))) == 1
    finally:
        QUALITY.clear();QUALITY.update(original)


def test_input_validation_and_normalized_blackbody_spectrum():
    with pytest.raises(ValueError):normalize_atmosphere({})
    a=default_atmosphere();a["aerosol_anisotropy"]=2.0
    with pytest.raises(ValueError):normalize_atmosphere(a)
    a=default_atmosphere();a["composition"]={"H2":1.1}
    with pytest.raises(ValueError):normalize_atmosphere(a)
    rgb=spectrum_rgb(6474.11)
    assert len(rgb)==3 and min(rgb)>0
    assert np.dot(rgb,[0.25,0.5,0.25]) == pytest.approx(1.0)
