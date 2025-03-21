"""Tests for analysis.py module."""

import pytest
from numpy.testing import assert_allclose
from packinganalysis import (PackingAnalysis,
                             read_shape_data, particle_volume)


# Define the fixture to initialize PackingAnalysis class for sphere
# packings
@pytest.fixture
def sphere_packing():
    shape_filepath = "src/packinganalysis/data/sphere/shape.csv"
    simulation_filepath = "src/packinganalysis/data/sphere/dump.csv"
    width = 20
    length = 20
    spheres = read_shape_data(shape_filepath)
    p_vol = particle_volume(spheres)
    packing = (PackingAnalysis(spheres, p_vol, simulation_filepath,
                               box_width=width,
                               box_length=length))
    return packing


# Define the fixture to initialize PackingAnalysis class for dimer(
# aspect ratio=2) packings
@pytest.fixture
def dimer_packing():
    shape_filepath = "src/packinganalysis/data/dimer/shape.csv"
    simulation_filepath = "src/packinganalysis/data/dimer/dump.csv"
    width = 20
    length = 20
    spheres = read_shape_data(shape_filepath)
    p_vol = particle_volume(spheres)
    packing = (PackingAnalysis(spheres, p_vol, simulation_filepath,
                               box_width=width,
                               box_length=length))
    return packing


# Tests for density and contact analysis of sphere packings
class TestPackingAnalysisSphere:
    @pytest.fixture(autouse=True)
    def setup_packing(self, sphere_packing):
        self.packing = sphere_packing

    def test_voronoi_density(self):
        density = self.packing.voronoi_density()
        assert_allclose(density, 0.64, atol=0.1)

    def test_centroid_density(self):
        density = self.packing.centroid_density(box_height=8)
        assert_allclose(density, 0.64, atol=0.1)

    def test_contact_analysis(self):
        contact_number, coordination_number = self.packing.contact_analysis()
        assert_allclose(contact_number, 6, atol=0.2)
        assert_allclose(coordination_number, 6, atol=0.2)


# Tests for density and contact analysis of sphere packings
class TestPackingAnalysisDimer:
    @pytest.fixture(autouse=True)
    def setup_packing(self, dimer_packing):
        self.packing = dimer_packing

    def test_voronoi_density(self):
        density = self.packing.voronoi_density()
        assert_allclose(density, 0.64, atol=0.1)

    def test_centroid_density(self):
        density = self.packing.centroid_density(box_height=8)
        assert_allclose(density, 0.64, atol=0.1)

    def test_contact_analysis(self):
        contact_number, coordination_number = self.packing.contact_analysis()
        assert_allclose(contact_number, 10, atol=0.3)
        assert_allclose(coordination_number, 8, atol=0.3)
