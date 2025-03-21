"""
Module for generating ghost particles due to periodic boundary
conditions and performing pairwise distance calculations to determine
contact, coordination and neighbor numbers.
"""

import pandas as pd
import numpy as np


def generate_ghost_particles(data: pd.DataFrame,
                             box_width: float = 20.0,
                             box_length: float = 20.0) -> pd.DataFrame:
    """This function generates ghost particles due to periodic boundary
    conditions in lateral directions(x-y). In periodic boundary conditions,
    a particle at the boundary appears on the opposite side of the boundary.
    To perform contact analysis and calculate pairwise distances, we need
    ghost particles.

    Parameters
    ----------
    data: pd.DataFrame
        The DataFrame that contains actual particles' information.
    box_width : float
        Width of simulation box. This is scaled by the diameter of the first
        sphere in the particle, which is set to 1 in the simulation.
    box_length : float
        Length of simulation box length. This is also scaled by the diameter
        of the first sphere in the particle.

    Returns
    -------
    extended_data: pd.DataFrame
        The DataFrame that contains both actual and ghost particles'
        information.

    """
    # Prepare a list to collect all the ghost and actual particles' data
    extended_data_list = []

    # Copy data and shift them along periodic boundaries
    for i in range(3):        # for x-axis
        for j in range(3):    # for y-axis
            ghost = data.copy()
            ghost['x'] += (i - 1) * box_width
            ghost['y'] += (j - 1) * box_length

            # Distinguish actual and ghost particles with "type"
            if i == 1 and j == 1:
                ghost['type'] = 'actual'
            else:
                ghost['type'] = 'ghost' + str(i) + str(j)

            # Append ghost Dataframe to the extended_data_list
            extended_data_list.append(ghost)

    # Concatenate all the DataFrames
    extended_data = pd.concat(extended_data_list, axis=0)

    return extended_data


def distance_calculations(actual_data: pd.DataFrame,
                          extended_data: pd.DataFrame,
                          contact: bool = True,
                          cutoff: float = 1.4) -> pd.DataFrame:
    """Performs pairwise distance calculations and detect particles in
    contact with or neighbor particles within a cutoff distance.

    To determine whether a particle has contact with another, pairwise
    distance calculations are performed per constituent spheres. Any pair
    of spheres belonging to different particles are in contact, then
    the particles are also considered as in contact. To determine a particle's
    neighbors within a given cutoff distance, distances are measured between
    the geometric centers of particles, not their constituent spheres.

    Parameters
    ----------
    actual_data : pd.DataFrame
        The DataFrame that contains actual particles' information.
    extended_data : pd.DataFrame
        The DataFrame that contains both actual and ghost particles'
        information, obtained from generate_ghost_particles function.
    contact : bool
        It indicates whether contact or neighbor calculations are in
        interest.
    cutoff : float
        The cutoff distance that are considered in neighbor calculations.

    Returns
    -------
    actual_data : pd.DataFrame
        The DataFrame that contains only actual particles, with added
        information: either ids and types of each particles' contacts or
        ids and types of each particles' neighbors within the given cutoff.
    """

    # Number of particles
    n_p = len(actual_data)

    # Convert actual data columns to numpy arrays
    actual_x = actual_data['x'].values
    actual_y = actual_data['y'].values
    actual_z = actual_data['z'].values
    ids = actual_data['id'].values

    # Convert extended data columns to numpy arrays
    extended_x = extended_data['x'].values
    extended_y = extended_data['y'].values
    extended_z = extended_data['z'].values
    extended_ids = extended_data['id'].values
    extended_types = extended_data['type'].values

    # Detect which particles are in contact, here the actual and extended
    # data information is given per constituent sphere
    if contact:
        # Convert radii columns to numpy arrays
        radii = actual_data['radius'].values
        extended_radii = extended_data['radius'].values

        # Store lists of ids and types of each constituent sphere's contacts
        contact_ids_list = []
        contact_types_list = []

        for i in range(n_p):
            # Calculate pairwise distances between each constituent sphere and
            # all the other spheres (including ghost ones)
            d_x = actual_x[i] - extended_x
            d_y = actual_y[i] - extended_y
            d_z = actual_z[i] - extended_z
            distances = np.sqrt(d_x * d_x + d_y * d_y + d_z * d_z)

            # Check if two spheres overlap or not
            cutoff = radii[i] + extended_radii

            # If they have the same id then it is not considered as contact
            mask = (distances <= cutoff) & (ids[i] != extended_ids)

            # Filter the contacts' ids and types
            contact_ids = extended_ids[mask]
            contact_types = extended_types[mask]

            # Append the results
            contact_ids_list.append(contact_ids.tolist())
            contact_types_list.append(contact_types.tolist())

        # Save contact_id and contact_type as columns in the actual data
        actual_data['contact_id'] = contact_ids_list
        actual_data['contact_type'] = contact_types_list

        return actual_data

    # Detect which particles are neighbors given in a cutoff distance,
    # here the actual and extended data information is given per particle
    else:
        # Store lists of ids and types of each particle's neighbors
        neighbor_ids_list = []
        neighbor_types_list = []

        for i in range(n_p):
            # Calculate pairwise distances between a particle's center and all
            # other particles' centers (including ghost ones)
            d_x = actual_x[i] - extended_x
            d_y = actual_y[i] - extended_y
            d_z = actual_z[i] - extended_z
            distances = np.sqrt(d_x * d_x + d_y * d_y + d_z * d_z)

            # Check if distances are within the cutoff distance,
            # exclude the reference particle itself
            mask = (distances <= cutoff) & (ids[i] != extended_ids)

            # Filter the neighbors' ids and types
            neighbor_ids = extended_ids[mask]
            neighbor_types = extended_types[mask]

            # Append the results
            neighbor_ids_list.append(neighbor_ids.tolist())
            neighbor_types_list.append(neighbor_types.tolist())

        # Save neighbor_id and neighbor_type as columns in the actual data
        actual_data['neighbor_id'] = neighbor_ids_list
        actual_data['neighbor_type'] = neighbor_types_list

        return actual_data
