"""Main module for the package 'packinganalysis'."""

import numpy as np
import math

from .data_preprocessing import DataPreprocessing
from .periodic_boundaries import (generate_ghost_particles,
                                  distance_calculations)
from .order_parameters import (SphericalHarmonics,
                               average_local_order_parameters,
                               global_order_parameters)
from itertools import chain
from collections import Counter


class PackingAnalysis(DataPreprocessing):
    """
    Analyse structural properties of disordered packings of mono-dispersed
    particles. The particle shape is defined by a number of overlapping
    spheres with varying radii.

    The packings are assumed to be periodic in the horizontal directions (x-y).
    This class calculates the following:
        - Packing density (with two methods: Voronoi and centroid methods),
        - Contact, coordination, and neighbor numbers,
        - Bond-orientational order parameters,
        - Radial distribution function.

    The particle shape details (spheres, p_vol) should be extracted from the
    shape file in particle_properties.py module, and the analysis is
    performed on preprocessed particle data provided by the simulation file.

    Parameters
    ----------
    spheres : list of tuple
        List of tuples representing the coordinates and radius of
        each constituent sphere.
    p_vol : float
        Volume of a particle.
    simulation_filepath : str
        Path to the file containing simulation data.

    Attributes
    ----------
    n_p : int
        Number of particles in the packing.
    max_diameter : float
        Maximum diameter of constituent spheres of the particle.
    z_min : float
        Minimum value of the z-coordinate of the bulk.
    z_max : float
        Maximum value of the z-coordinate of the bulk.
    bulk_mask : np.ndarray of bool
        Boolean values for filtering bulk particles.
    """

    def __init__(self, spheres: list, p_vol: float,
                 simulation_filepath: str,
                 box_width: float, box_length: float) -> None:
        """Initialize the PackingAnalysis class with particle shape properties
        and simulation data.

        This constructor assigns shape information and reads simulation
        files, then processes the data to calculate basic properties needed
        for further analysis.

        Parameters
        ----------
        spheres : list of tuple
            List of tuples representing the coordinates and radius of
            each constituent sphere.
        p_vol : float
            Volume of a particle.
        simulation_filepath : str
            Path to the file containing simulation data.
        """

        super().__init__(simulation_filepath, box_width, box_length)
        self.spheres = spheres
        self.p_vol = p_vol
        self.n_p = self.particle_data.shape[0]
        self.max_diameter = 2 * max([sphere[3] for sphere in self.spheres])
        self.z_min = self.sphere_data['z'].min() + 5 * self.max_diameter
        self.z_max = self.sphere_data['z'].max() - 5 * self.max_diameter
        self.bulk_mask = ((self.particle_data['z'] > self.z_min)
                          & (self.particle_data['z'] < self.z_max))

    def voronoi_density(self) -> float:
        """Measure packing density by using Voronoi volumes of particles.

        Voronoi volume of a particle collects all the points closest to the
        particle than any other particle in the packing. Summing Voronoi
        volumes of all the bulk particles gives us the total bulk volume.
        The packing density is then obtained by dividing the volume
        occupied by the bulk particles to the total bulk volume.

        Returns
        -------
        voronoi_fi : float
            Packing density measured by Voronoi method.
        """

        # Determine number of bulk particles and bulk volume
        bulk_particles = self.particle_data[self.bulk_mask]
        n_bulk = len(bulk_particles)
        bulk_vol = bulk_particles['Voronoi_volume'].sum()

        # Packing density calculation
        occupied_vol = n_bulk * self.p_vol
        voronoi_fi = occupied_vol / bulk_vol

        return voronoi_fi

    def centroid_density(self, box_height: float = 8.0, steps: int = 50) -> (
                         float):
        """Measure packing density by using the centroid method.

        This method calculates the packing density by considering particles
        whose centroids fall within a fixed rectangular bulk region. The
        region is defined by the parameters `box_width`, `box_length`,
        and `box_height`. Starting from the lower boundary (`self.z_min`),
        the region is shifted stepwise along the z-direction, and packing
        density calculations are performed at each step. The final packing
        density is averaged over all steps.

        Parameters
        ----------
        box_height : float
            Height of the bulk region. This is also scaled by the diameter
            of the first sphere in the particle.
             (Default value = 8.0)
        steps : int
            Number of steps for shifting the bulk region along the
            z-direction. At each step, packing density is
            calculated and averaged over all steps.
             (Default value = 50)

        Returns
        -------
        centroid_fi : float
            The packing density measured by the Centroid method, averaged
            over all steps.
        """
        # Determine the bulk volume
        bulk_vol = self.box_width * self.box_length * box_height

        # Set the starting and stopping levels
        start_l = self.z_min
        stop_l = self.particle_data['z'].max()

        # Determine shifting size of the bulk region in the z-direction
        shift_size = (stop_l - start_l - box_height) / steps

        # Variable to store packing density at each step
        centroid_fis = []

        # Centroid method
        for i in range(steps):

            # Set the boundaries
            lower_b = start_l + shift_size * i
            upper_b = lower_b + box_height

            # Filter particles within the bulk
            bulk_particles = self.particle_data[
                (self.particle_data['z'] >= lower_b) &
                (self.particle_data['z'] <= upper_b)]

            # Determine number of bulk particles
            n_bulk = len(bulk_particles)

            # Packing density calculation
            occupied_vol = n_bulk * self.p_vol
            fi = occupied_vol / bulk_vol
            centroid_fis.append(fi)

        # Compute the average packing density
        centroid_fi = np.array(centroid_fis).mean()

        return centroid_fi

    def contact_analysis(self) -> (float, float):
        """Performs contact analysis in the packing.

        Determines contact number and coordination number per particle.
        Coordination number of a particle is number of neighbouring
        particles having at least one contact with that particle.

        Returns
        -------
        contact_number : float
            Average contact number in the bulk
        coordination_number : float
            Average coordination number in the bulk
        """
        # Generate ghost particles due to periodic boundaries
        extended_data = generate_ghost_particles(self.sphere_data,
                                                 self.box_width,
                                                 self.box_length)

        # Find ids and types of contacts for each constituent sphere
        self.sphere_data = distance_calculations(self.sphere_data,
                                                 extended_data,
                                                 contact=True)

        # Find ids and types of contacts per particle from its
        # constituent spheres' contacts
        columns = ['contact_id', 'contact_type']
        for col in columns:
            groups = self.sphere_data.groupby('id')[col].agg(
                lambda x: list(chain(*x)))
            self.particle_data[col] = self.particle_data['id'].map(groups)

        # Calculate contact number per particle
        self.particle_data['contact_number'] = self.particle_data[
            'contact_id'].map(len)

        # Calculate coordination number per particle
        self.particle_data['coordination_number'] = self.particle_data[
            'contact_id'].map(lambda x: len(set(x)))

        # Extract bulk particles
        bulk_particles = self.particle_data[self.bulk_mask]

        # Calculate average contact and coordination numbers
        contact_number = bulk_particles['contact_number'].mean()
        coordination_number = bulk_particles['coordination_number'].mean()

        return float(contact_number), float(coordination_number)

    def neighbor_analysis(self, shell_d: float = 1.4) -> float:
        """Perform neighboring analysis in the packing.

        Determines neighbor number per particle given a cutoff distance.
        The cutoff distance is obtained by multiplying self.max_diameter
        and shell_d.

        Parameters
        ----------
        shell_d : float
            Shell thickness to include neighbors

        Returns
        -------
        neighbor_number : float
            Average neighbor number in the bulk
        """
        # Calculate cutoff distance to find neighbors
        cutoff = self.max_diameter * shell_d

        # Generate ghost particles due to periodic boundaries
        extended_data = generate_ghost_particles(self.particle_data,
                                                 self.box_width,
                                                 self.box_length)

        # Find neighbors of each particle within the cutoff distance
        self.particle_data = (distance_calculations(self.particle_data,
                                                    extended_data,
                                                    contact=False,
                                                    cutoff=cutoff))

        # Calculate neighbor number per particle
        self.particle_data['neighbor_number'] = self.particle_data[
            'neighbor_id'].map(lambda x: len(set(x)))

        # Extract bulk particles
        bulk_particles = self.particle_data[self.bulk_mask]

        # Calculate average contact and coordination numbers
        neighbor_number = bulk_particles['neighbor_number'].mean()

        return float(neighbor_number)

    def bond_orientational_order_parameters(self, ordered: bool = False,
                                            shell_d: float = 1.4) -> (
                                            (tuple, tuple)):
        """Calculates bond-orientational order parameters to determine local
        and global structures in the packing.

        If the packing is disordered, only particles are in contact with
        will be considered in the order parameter calculations. If it is
        ordered, then the neighbor particles within the given cutoff
        distance will be considered.

        Parameters
        ----------
        ordered : bool
            True if the packing is ordered, False if it is disordered
        shell_d : float
            Shell thickness to include neighbors

        Returns
        -------
        local_order_tuple : tuple of float
            A tuple of local order parameters: q4, q6, q4_tilda_bar,
            q6_tilda_bar
        global_order_tuple : tuple of float
            A tuple of global order parameters: global_q4, global_q6
        """
        # Packing is ordered
        if ordered:
            # Find neighbors of each particle within given the cutoff
            self.neighbor_analysis(shell_d)

            # Extract lists of 'neighbor_id' and 'neighbor_type' of particles
            neighbor_ids = self.particle_data['neighbor_id'].values
            neighbor_types = self.particle_data['neighbor_type'].values

        # Packing is disordered
        else:
            # Find neighbors of each particle that are in contact with
            self.contact_analysis()

            # Extract lists of 'neighbor_id' and 'neighbor_type' of particles
            neighbor_ids = self.particle_data['contact_id'].values
            neighbor_types = self.particle_data['contact_type'].values

        # Convert neighbor ids and types into a list of tuples for all rows
        neighbor_tuples = [list(zip(ids, types)) for ids, types in
                           zip(neighbor_ids, neighbor_types)]

        # Convert the neighbor tuples to a set for efficient lookup
        neighbor_sets = [set(tuples) for tuples in neighbor_tuples]
        neighbor_ids = [set(ids) for ids in neighbor_ids]

        # Generate ghost particles due to periodicity
        extended_data = generate_ghost_particles(self.particle_data)

        # Convert ids and types of extended data into tuples
        extended_data['tuple'] = list(zip(extended_data['id'], extended_data[
            'type']))

        # Convert actual data columns to numpy arrays
        actual_x = self.particle_data['x'].values
        actual_y = self.particle_data['y'].values
        actual_z = self.particle_data['z'].values

        # Store spherical harmonics qm of degrees 2, 4, 6 and corresponding
        # local order parameters q
        q2m_r_list, q4m_r_list, q6m_r_list = [], [], []
        q2m_hat_list, q4m_hat_list, q6m_hat_list = [], [], []
        q2_list, q4_list, q6_list = [], [], []

        # Calculate spherical harmonics and local order parameters
        for i in range(self.n_p):
            # Extract neighbor set of particle i
            neighbor_set = neighbor_sets[i]

            # Filter neighbor data of particle i
            mask = extended_data['tuple'].isin(neighbor_set)
            neighbor_data = extended_data[mask]

            # Convert neighbor data columns to numpy arrays
            neighbor_x = neighbor_data['x'].values
            neighbor_y = neighbor_data['y'].values
            neighbor_z = neighbor_data['z'].values

            # Calculate pairwise distances between a particle's center and all
            # other particles' centers (including ghost ones)
            d_x = actual_x[i] - neighbor_x
            d_y = actual_y[i] - neighbor_y
            d_z = actual_z[i] - neighbor_z

            distances = np.sqrt(d_x * d_x + d_y * d_y + d_z * d_z)

            # Call the class SphericalHarmonics
            local_order_parameters = SphericalHarmonics(d_x, d_y, d_z,
                                                        distances)

            # Calculate spherical harmonics and local order parameters
            q2m_r, q2m, q2m_hat, q2 = local_order_parameters.q2()
            q4m_r, q4m, q4m_hat, q4 = local_order_parameters.q4()
            q6m_r, q6m, q6m_hat, q6 = local_order_parameters.q6()

            # Append the results to the corresponding lists
            q2m_r_list.append(q2m_r)
            q4m_r_list.append(q4m_r)
            q6m_r_list.append(q6m_r)

            q2_list.append(q2)
            q4_list.append(q4)
            q6_list.append(q6)

            q2m_hat_list.append(q2m_hat)
            q4m_hat_list.append(q4m_hat)
            q6m_hat_list.append(q6m_hat)

        # Update particle_data with spherical harmonics and local order
        # parameters per particle
        self.particle_data['q2m_r'] = q2m_r_list
        self.particle_data['q4m_r'] = q4m_r_list
        self.particle_data['q6m_r'] = q6m_r_list

        self.particle_data['q2'] = q2_list
        self.particle_data['q4'] = q4_list
        self.particle_data['q6'] = q6_list

        self.particle_data['q2m_hat'] = q2m_hat_list
        self.particle_data['q4m_hat'] = q4m_hat_list
        self.particle_data['q6m_hat'] = q6m_hat_list

        self.particle_data = average_local_order_parameters(
            self.particle_data, neighbor_ids)

        # Extract bulk particles
        bulk_particles = self.particle_data[self.bulk_mask]

        # Calculate bulk-averaged local order parameters
        q4 = float(bulk_particles['q4'].mean())
        q6 = float(bulk_particles['q6'].mean())

        # Calculate bulk-averaged updated local order parameters
        q4_tilda_bar = float(bulk_particles['q4_tilda_bar'].mean())
        q6_tilda_bar = float(bulk_particles['q6_tilda_bar'].mean())

        # Create a tuple of bulk-averaged local order parameters
        local_order_tuple = (q4, q6, q4_tilda_bar, q6_tilda_bar)

        # Calculate sum of neighbor number of each particle in the bulk
        bulk_neighbor_ids = [neighbor_ids[i] for i in bulk_particles.index]
        bulk_n_neighbor = sum([len(neighbor_id) for neighbor_id in
                               bulk_neighbor_ids])

        # Calculate global order parameters
        global_q2, global_q4, global_q6 = global_order_parameters(
            bulk_particles, bulk_n_neighbor)

        # Create a tuple of bulk-averaged global order parameters
        global_order_tuple = (float(global_q4), float(global_q6))

        return local_order_tuple, global_order_tuple

    def radial_distribution_function(self, shell_width: float = 0.025) -> (
                                     np.ndarray):
        """Calculates radial distribution function (rdf) with given shell
        thickness.

        Parameters
        ----------
        shell_width : float
            Shell thickness considered in the rdf calculations

        Returns
        -------
        rdf : np.ndarray
            An array of radial distribution function results, the first column
            is distance/max_diameter
        """

        # Maximum distance in the rdf calculations are the half of the
        # box_width due to periodic boundaries.
        max_distance_periodic = self.box_width / 2

        # The length of the resulting array of rdf
        rdf_length = int(max_distance_periodic / shell_width)

        # Initiate array of rdf
        rdf = np.zeros((rdf_length, 2))
        rdf[:, 0] = np.arange(0, rdf_length) * shell_width

        # Generate ghost particles for distance calculations
        extended_data = generate_ghost_particles(self.particle_data,
                                                 self.box_width,
                                                 self.box_length)

        # Extract bulk data
        bulk_actual_data = self.particle_data[self.bulk_mask]

        # Determine number of bulk particles
        n_bulk = len(bulk_actual_data)

        # Convert actual data columns to numpy arrays
        actual_x = bulk_actual_data['x'].values
        actual_y = bulk_actual_data['y'].values
        actual_z = bulk_actual_data['z'].values

        # Convert extended data columns to numpy arrays
        extended_x = extended_data['x'].values
        extended_y = extended_data['y'].values
        extended_z = extended_data['z'].values

        # Rdf calculation
        for i in range(n_bulk):
            # Calculate distances between each bulk particle and all
            # the other particles (including ghost ones)
            d_x = actual_x[i] - extended_x
            d_y = actual_y[i] - extended_y
            d_z = actual_z[i] - extended_z
            distances = np.sqrt(d_x * d_x + d_y * d_y + d_z * d_z)

            # Determine corresponding shell of the calculated distance
            shells = distances / shell_width
            int_shells = shells.astype(int)

            # Filter shells due to periodicity
            mask = distances <= max_distance_periodic

            # Add 1 in the corresponding shells
            int_shells_count = Counter(int_shells[mask])
            unique_int_shells = list(int_shells_count.keys())
            n_particles_in_shells = list(int_shells_count.values())
            rdf[unique_int_shells, 1] += n_particles_in_shells

        # Delete the first row
        rdf = np.delete(rdf, 0, axis=0)

        # Determine particle number density
        rho = n_bulk / (self.box_width * self.box_length
                        * (self.z_max - self.z_min))

        # Determine volume of the shell
        v_shell = 4 * math.pi * shell_width * rdf[:, 0] ** 2

        # Determine constant factor in the rdf calculations
        constant_factor = n_bulk * v_shell * rho

        # Scale the second column
        rdf[:, 1] /= constant_factor

        return rdf
