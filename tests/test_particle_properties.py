"""Tests for `packinganalysis` package, particle_properties.py module."""

import numpy as np
import pytest
from numpy.ma.testutils import assert_almost_equal
from numpy.testing import assert_allclose

from packinganalysis.particle_properties import (particle_volume,
                                                 volume_com_moment,
                                                 point_contains)


# Define the fixture to store properties of sphere of radius=0.5
@pytest.fixture
def sphere():
    spheres = [(0, 0, 0, 0.5)]
    p_volume = 0.5235987755982988
    com = [0, 0, 0]
    m_inertias = [0.1, 0.1, 0.1, 0, 0, 0]
    return spheres, p_volume, com, m_inertias


# Define the fixture to store properties of dimer of aspect ratio=2 composed
# of two constituent spheres of radius=0.5
@pytest.fixture
def dimer_aspect_ratio2():
    spheres = [(0, 0, 0, 0.5), (0, 0, 1, 0.5)]
    p_volume = 2 * 0.5235987755982988
    com = [0, 0, 0.5]
    m_inertias = [0.35, 0.35, 0.1, 0, 0, 0]
    return spheres, p_volume, com, m_inertias


# Tests for sphere properties
class TestSphere:
    @pytest.fixture(autouse=True)
    def setup_sphere(self, sphere):
        self.spheres, self.p_volume, self.com, self.m_inertias = sphere

    def test_particle_volume(self):
        assert particle_volume(self.spheres) == self.p_volume

    def test_volume_com_moment(self):
        result = volume_com_moment(self.spheres, 1000000)
        assert_almost_equal(result[0], self.p_volume, decimal=2)
        assert_allclose(result[1], self.com, atol=0.001)
        assert_allclose(result[2], self.m_inertias, atol=0.001)


# Tests for dimer properties
class TestDimer:
    @pytest.fixture(autouse=True)
    def setup_dimer(self, dimer_aspect_ratio2):
        self.spheres, self.p_volume, self.com, self.m_inertias = (
            dimer_aspect_ratio2)

    def test_particle_volume(self):
        assert particle_volume(self.spheres) == self.p_volume

    def test_volume_com_moment(self):
        result = volume_com_moment(self.spheres, 1000000)
        assert_almost_equal(result[0], self.p_volume, decimal=2)
        assert_allclose(result[1], self.com, atol=0.001)
        assert_allclose(result[2], self.m_inertias, atol=0.001)


def test_point_contains():
    assert point_contains(np.array([0, 0, 0]), (0, 0, 0, 0.5))
    assert not point_contains(np.array([2, 2, 2]), (0, 0, 0, 0.5))
