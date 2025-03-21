"""Module for calculating bond-orientational order parameters."""

import numpy as np
import pandas as pd
import math
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class SphericalHarmonics:
    """Calculate local bond-orientational order parameters in terms
    of spherical harmonics as defined by Steinhardt et al.

    Parameters
    ----------
    x :  np.ndarray
        A numpy array of separations of a particle's x coordinate and its
        neighbors' x coordinates
    y :  np.ndarray
        A numpy array of separations of a particle's y coordinate and its
        neighbors' y coordinates
    z :  np.ndarray
        A numpy array of separations of a particle's z coordinate and its
        neighbors' z coordinates
    distances :  np.ndarray
        A numpy array of distances between a particle and its neighbors

    Attributes
    ----------
    n_neighbor : int
        Neighbor number of a particle considered for the order parameter
        calculations.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 distances: np.ndarray) -> None:
        """This constructor initiates the class SphericalHarmonics.
        Computes arrays to be used in the spherical harmonics calculations.

        Parameters
        ----------
        x :  np.ndarray
            A numpy array of separations of a particle's x coordinate and its
            neighbors' x coordinates
        y :  np.ndarray
            A numpy array of separations of a particle's y coordinate and its
            neighbors' y coordinates
        z :  np.ndarray
            A numpy array of separations of a particle's z coordinate and its
            neighbors' z coordinates
        distances :  np.ndarray
            A numpy array of distances between a particle and its neighbors
        """
        self.x = x
        self.y = y
        self.z = z
        self.distances = distances
        self.n_neighbor = len(self.distances)

        # Precompute array values
        self.complex_xy = self.x + 1j * self.y
        self.complex_x_y = self.x + 1j * (-self.y)

        # Power of arrays to be used in next calculations
        self.pow6_complex_xy = self.complex_xy ** 6
        self.pow5_complex_xy = self.complex_xy ** 5
        self.pow4_complex_xy = self.complex_xy ** 4
        self.pow3_complex_xy = self.complex_xy ** 3
        self.pow2_complex_xy = self.complex_xy ** 2

        self.pow6_complex_x_y = self.complex_x_y ** 6
        self.pow5_complex_x_y = self.complex_x_y ** 5
        self.pow4_complex_x_y = self.complex_x_y ** 4
        self.pow3_complex_x_y = self.complex_x_y ** 3
        self.pow2_complex_x_y = self.complex_x_y ** 2

        self.pow4_z = self.z ** 4
        self.pow2_z = self.z ** 2

        self.pow6_distances = self.distances ** 6
        self.pow4_distances = self.distances ** 4
        self.pow2_distances = self.distances ** 2

        self.pow6_z_distances = (self.z / self.distances) ** 6
        self.pow4_z_distances = (self.z / self.distances) ** 4
        self.pow2_z_distances = (self.z / self.distances) ** 2

    def q2(self) -> (np.ndarray, np.ndarray, np.ndarray, float):
        """Calculates spherical harmonics of degree 2 and determines
        rotational invariant of them.

        Returns
        -------
        q2m_r : np.ndarray
            Spherical harmonics of degree=2
        q2m : np.ndarray
            Average spherical harmonics of degree=2
        q2m_hat : np.ndarray
            Scaled average spherical harmonics of degree=2
        q2 : np.ndarray
            Rotational invariant of spherical harmonics of degree=2
            (Local order parameter)
        """
        # Store spherical harmonics of degree 2 in an array
        q2m_r = np.zeros(5, dtype=complex)

        # Precompute reusable values
        sqrt_7_5_pi = math.sqrt(7.5 / math.pi)
        sqrt_5_pi = math.sqrt(5 / math.pi)

        # Calculate spherical harmonics
        q2m_r[0] = np.sum(1 / 4 * sqrt_7_5_pi * self.pow2_complex_x_y
                          / self.pow2_distances)
        q2m_r[1] = np.sum(1 / 2 * sqrt_7_5_pi * self.complex_x_y * self.z
                          / self.pow2_distances)
        q2m_r[2] = np.sum(1 / 4 * sqrt_5_pi * (3 * self.pow2_z
                          - self.pow2_distances) / self.pow2_distances)
        q2m_r[3] = np.sum(-1 / 2 * sqrt_7_5_pi * self.complex_xy * self.z
                          / self.pow2_distances)
        q2m_r[4] = np.sum(1 / 4 * sqrt_7_5_pi * self.pow2_complex_xy
                          / self.pow2_distances)

        # Take average of spherical harmonics
        q2m = np.where((self.n_neighbor > 0) & (~np.isnan(q2m_r)), q2m_r
                       / self.n_neighbor, np.zeros_like(q2m_r))

        # Scale q2m
        q2m_hat = scale_qm(q2m)

        # Construct rotational invariant of spherical harmonics
        q2 = math.sqrt(4 * math. pi / 5 * np.sum(abs(q2m) ** 2))

        return q2m_r, q2m, q2m_hat, q2

    def q4(self) -> (np.ndarray, np.ndarray, np.ndarray, float):
        """Calculates spherical harmonics of degree 4 and determines
        rotational invariant of them.

        Returns
        -------
        q4m_r : np.ndarray
            Spherical harmonics of degree=4
        q4m : np.ndarray
            Average spherical harmonics of degree=4
        q4m_hat : np.ndarray
            Scaled average spherical harmonics of degree=4
        q4 : np.ndarray
            Rotational invariant of spherical harmonics of degree=4
            (Local order parameter)
        """
        # Store spherical harmonics of degree 4 in an array
        q4m_r = np.zeros(9, dtype=complex)

        # Precompute reusable values
        sqrt_35_pi = math.sqrt(35 / math.pi)
        sqrt_17_5_pi = math.sqrt(17.5 / math.pi)
        sqrt_5_pi = math.sqrt(5 / math.pi)
        sqrt_2_5_pi = math.sqrt(2.5 / math.pi)
        sqrt_1_pi = math.sqrt(1 / math.pi)

        # Calculate spherical harmonics
        q4m_r[0] = np.sum(3 / 16 * sqrt_17_5_pi * self.pow4_complex_x_y
                          / self.pow4_distances)
        q4m_r[1] = np.sum(3 / 8 * sqrt_35_pi * self.pow3_complex_x_y * self.z
                          / self.pow4_distances)
        q4m_r[2] = np.sum(3 / 8 * sqrt_2_5_pi * self.pow2_complex_x_y
                          * (7 * self.pow2_z - self.pow2_distances)
                          / self.pow4_distances)
        q4m_r[3] = np.sum(3 / 8 * sqrt_5_pi * self.complex_x_y * self.z
                          * (7 * self.pow2_z - 3 * self.pow2_distances)
                          / self.pow4_distances)
        q4m_r[4] = np.sum(3 / 16 * sqrt_1_pi * (35 * self.pow4_z - 30
                          * self.pow2_z * self.pow2_distances + 3 *
                          self.pow4_distances) / self.pow4_distances)
        q4m_r[5] = np.sum(-3 / 8 * sqrt_5_pi * self.complex_xy * self.z
                          * (7 * self.pow2_z - 3 * self.pow2_distances)
                          / self.pow4_distances)
        q4m_r[6] = np.sum(3 / 8 * sqrt_2_5_pi * self.pow2_complex_xy
                          * (7 * self.pow2_z - self.pow2_distances)
                          / self.pow4_distances)
        q4m_r[7] = np.sum(-3 / 8 * sqrt_35_pi * self.pow3_complex_xy * self.z
                          / self.pow4_distances)
        q4m_r[8] = np.sum(3 / 16 * sqrt_17_5_pi * self.pow4_complex_xy
                          / self.pow4_distances)

        # Take average of spherical harmonics
        q4m = np.where((self.n_neighbor > 0) & (~np.isnan(q4m_r)), q4m_r
                       / self.n_neighbor, np.zeros_like(q4m_r))

        # Scale q4m
        q4m_hat = scale_qm(q4m)

        # Construct rotational invariant of spherical harmonics
        q4 = math.sqrt(4 * math. pi / 9 * np.sum(abs(q4m) ** 2))

        return q4m_r, q4m, q4m_hat, q4

    def q6(self) -> (np.ndarray, np.ndarray, np.ndarray, float):
        """Calculates spherical harmonics of degree 6 and determines
        rotational invariant of them.

        Returns
        -------
        q6m_r : np.ndarray
            Spherical harmonics of degree=6
        q6m : np.ndarray
            Average spherical harmonics of degree=6
        q6m_hat : np.ndarray
            Scaled average spherical harmonics of degree=6
        q6 : np.ndarray
            Rotational invariant of spherical harmonics of degree=6
            (Local order parameter)
        """
        q6m_r = np.zeros(13, dtype=complex)

        # Precompute reusable values
        sqrt_3003_pi = math.sqrt(3003 / math.pi)
        sqrt_1365_pi = math.sqrt(1365 / math.pi)
        sqrt_1001_pi = math.sqrt(1001 / math.pi)
        sqrt_136_5_pi = math.sqrt(136.5 / math.pi)
        sqrt_45_5_pi = math.sqrt(45.5 / math.pi)
        sqrt_13_pi = math.sqrt(13 / math.pi)

        # Calculate spherical harmonics
        q6m_r[0] = np.sum(1 / 64 * sqrt_3003_pi * self.pow6_complex_x_y
                          / self.pow6_distances)
        q6m_r[1] = np.sum(3 / 32 * sqrt_1001_pi * self.pow5_complex_x_y
                          * self.z / self.pow6_distances)
        q6m_r[2] = np.sum(3 / 32 * sqrt_45_5_pi * self.pow4_complex_x_y
                          * (11 * self.pow2_z - self.pow2_distances)
                          / self.pow6_distances)
        q6m_r[3] = np.sum(1 / 32 * sqrt_1365_pi * self.pow3_complex_x_y *
                          self.z * (11 * self.pow2_z - 3 * self.pow2_distances)
                          / self.pow6_distances)
        q6m_r[4] = np.sum(1 / 64 * sqrt_1365_pi * self.pow2_complex_x_y
                          * (33 * self.pow4_z - 18 * self.pow2_distances
                             * self.pow2_z + self.pow4_distances)
                          / self.pow6_distances)
        q6m_r[5] = np.sum(1 / 16 * sqrt_136_5_pi * self.complex_x_y * self.z
                          * (33 * self.pow4_z - 30 * self.pow2_distances
                             * self.pow2_z + 5 * self.pow4_distances)
                          / self.pow6_distances)
        q6m_r[6] = np.sum(1 / 32 * sqrt_13_pi * (231 * self.pow6_z_distances
                          - 315 * self.pow4_z_distances
                          + 105 * self.pow2_z_distances - 5))
        q6m_r[7] = np.sum(-1 / 16 * sqrt_136_5_pi * self.complex_xy * self.z
                          * (33 * self.pow4_z - 30 * self.pow2_distances
                             * self.pow2_z + 5 * self.pow4_distances)
                          / self.pow6_distances)
        q6m_r[8] = np.sum(1 / 64 * sqrt_1365_pi * self.pow2_complex_xy
                          * (33 * self.pow4_z - 18 * self.pow2_distances
                             * self.pow2_z + self.pow4_distances)
                          / self.pow6_distances)
        q6m_r[9] = np.sum(-1 / 32 * sqrt_1365_pi * self.pow3_complex_xy *
                          self.z * (11 * self.pow2_z - 3 * self.pow2_distances)
                          / self.pow6_distances)
        q6m_r[10] = np.sum(3 / 32 * sqrt_45_5_pi * self.pow4_complex_xy
                           * (11 * self.pow2_z - self.pow2_distances)
                           / self.pow6_distances)
        q6m_r[11] = np.sum(-3 / 32 * sqrt_1001_pi * self.pow5_complex_xy
                           * self.z / self.pow6_distances)
        q6m_r[12] = np.sum(1 / 64 * sqrt_3003_pi * self.pow6_complex_xy
                           / self.pow6_distances)

        # Take average of spherical harmonics
        q6m = np.where((self.n_neighbor > 0) & (~np.isnan(q6m_r)), q6m_r
                       / self.n_neighbor, np.zeros_like(q6m_r))

        # Scale q6m
        q6m_hat = scale_qm(q6m)

        # Construct rotational invariant of spherical harmonics
        q6 = math.sqrt(4 * math. pi / 13 * np.sum(abs(q6m) ** 2))

        return q6m_r, q6m, q6m_hat, q6


def scale_qm(qm: np.ndarray) -> np.ndarray:
    """Scale qm array.
    Parameters
    ----------
    qm : np.ndarray
        Average spherical harmonics

    Returns
    -------
    qm_hat :  np.ndarray
        Scaled average spherical harmonics
    """
    total_qm = math.sqrt(np.sum(abs(qm) ** 2))

    qm_hat = np.where((total_qm > 0) & (~np.isnan(total_qm)),
                      qm / total_qm, np.zeros_like(qm))

    return qm_hat


def average_local_order_parameters(data: pd.DataFrame,
                                   neighbor_ids: list) -> pd.DataFrame:
    """Calculates local order parameters defined by Eslami et al. to
    improve the accuracy of crystal structure determination.

    Parameters
    ----------
    data : pd.DataFrame
        This DataFrame inherits the results of the class SphericalHarmonics
    neighbor_ids: list of int
        This list contains neighbour ids of particles

    Returns
    -------
    data : pd.DataFrame
        Updated DataFrame includes the calculated local order parameters here
    """
    # Number of particles in the data
    n_p = len(data)

    # Store the q_tilda values of different degrees 2, 4, 6
    q_tildas = {'q2m_hat': [], 'q4m_hat': [], 'q6m_hat': []}

    for i in range(n_p):
        # Extract neighbor_ids of particle i
        neighbor_id = neighbor_ids[i]

        # Filter neighbor data for particle i
        mask = data['id'].isin(neighbor_id)
        neighbor_data = data[mask]

        # Calculate dot products between qm_hat arrays of particle i
        # and its neighbors
        for key, value in q_tildas.items():
            q_tilda = neighbor_data[key].apply(
                lambda x: np.vdot(data.loc[i, key], x).real).mean()
            value.append(q_tilda)

    # Assign the averaged dot products to data's columns
    data['q2_tilda'] = q_tildas['q2m_hat']
    data['q4_tilda'] = q_tildas['q4m_hat']
    data['q6_tilda'] = q_tildas['q6m_hat']

    q_tilda_bars = {'q2_tilda': [], 'q4_tilda': [], 'q6_tilda': []}

    for i in range(n_p):
        # Extract neighbor_ids of particle i and measure its neighbor
        # number
        neighbor_id = neighbor_ids[i]
        n_neighbor = len(neighbor_id)

        # Filter neighbor data for particle i
        mask = data['id'].isin(neighbor_id)
        neighbor_data = data[mask]

        # Calculate new local order parameters of particle i by averaging
        # q_tildas over its neighbours
        for key, value in q_tilda_bars.items():
            q_tilda_bar = data.loc[i, key] + neighbor_data[key].sum()
            value.append(q_tilda_bar / (n_neighbor + 1))

    data['q2_tilda_bar'] = q_tilda_bars['q2_tilda']
    data['q4_tilda_bar'] = q_tilda_bars['q4_tilda']
    data['q6_tilda_bar'] = q_tilda_bars['q6_tilda']

    return data


def global_order_parameters(data: pd.DataFrame,
                            total_n_neighbor: int) -> (float, float, float):
    """Calculates global bond-orientational order parameters.

    Parameters
    ----------
    data : pd.DataFrame
        This DataFrame inherits the results of the class SphericalHarmonics
    total_n_neighbor: int
        Sum of neighbour numbers of all the particles.

    Returns
    -------
    global_q2 : float
        Global order parameter of degree=2
    global_q4 : float
        Global order parameter of degree=4
    global_q6 : float
        Global order parameter of degree=6
    """
    # Sum spherical harmonics for each degree
    total_qm_r = data[['q2m_r', 'q4m_r', 'q6m_r']].sum()

    # Store global order parameters: global_q2, global_q4, global_q6
    global_qs = {'q2m_r': 0, 'q4m_r': 0, 'q6m_r': 0}

    # Calculate globals
    for key in global_qs.keys():
        average_qm_r = total_qm_r[key] / total_n_neighbor
        global_q = math.sqrt(4 * math.pi / (2 * int(key[1]) + 1) * np.sum(abs(
            average_qm_r) ** 2))
        global_qs[key] = global_q

    # Extract global order parameters
    global_q2 = global_qs['q2m_r']
    global_q4 = global_qs['q4m_r']
    global_q6 = global_qs['q6m_r']

    return global_q2, global_q4, global_q6
