"""
Module for calculating volume, center of mass and moment of inertia
of particle shapes composed of a varied number of overlapping spheres.
"""

import numpy as np
import math
import csv


def read_shape_data(shape_filepath: str) -> list:
    """Read the particle shape data from a .csv file the given location.

    The shape file is expected to have four columns (coordinates: x, y,
    z, and radius), where each row corresponds to a constituent sphere
    of the particle.

    Parameters
    ----------
    shape_filepath : str
        Path to the file containing particle shape data.

    Returns
    -------
    spheres : list of tuple
        List of tuples representing the coordinates and radius of
        each constituent sphere.

    """

    # List to store information about the constituent spheres of the
    # particle
    spheres = []

    # Read the particle shape data file
    with open(shape_filepath, "r", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 4:
                try:
                    x, y, z, radius = map(float, row)
                    spheres.append((x, y, z, radius))
                except ValueError:
                    print(f"Warning: Skipping invalid line {row}.")
            else:
                print("Data formatting is not acceptable.")
    return spheres


def particle_volume(spheres: list,
                    n_points: int = 100000,
                    n_simulations: int = 5,
                    mass: float = 1.0) -> float:
    """Calculate volume of a particle composed of one or more spheres.

    The coordinates and radius information of the constituent spheres are
    provided in 'spheres' list. Except sphere and symmetric dimer,
    this function calls another function(volume_com_moment) to calculate
    volume of more complicated shapes with Monte-Carlo method.

    Parameters
    ----------
    spheres: list of tuple
        List of tuples representing the coordinates and radius of each
        constituent sphere.
    n_points: int, optional
        The number of points used in each Monte-Carlo simulation to
        estimate these properties. A higher number improves
        accuracy at the cost of performance.
         (Default value = 100000)
    n_simulations: int, optional
        The number of times a Monte-Carlo simulation is run.
         (Default value = 5)
    mass: float, optional
        The unit mass of a sphere.
         (Default value = 1.0)

    Returns
    -------
    p_volume: float
        Volume of the particle shape.
    """

    # Check how many spheres the particle shape includes
    if len(spheres) == 1:
        # Calculate the volume of a sphere
        _, _, _, radius = spheres[0]
        p_volume = 4 / 3 * math.pi * radius**3
        print("The particle shape is sphere.")

    elif len(spheres) == 2:
        # Check if it is a symmetric dimer
        x1, y1, z1, radius1 = spheres[0]
        x2, y2, z2, radius2 = spheres[1]

        if radius1 == radius2:
            diameter = 2 * radius1
            # Calculate major length and aspect ratio of the symmetric dimer
            major_length = (max(abs(x2 - x1), abs(y2 - y1), abs(z2 - z1))
                            + diameter)
            aspect_ratio = major_length / diameter

            # Calculate overlapping volume of the two spheres
            h1 = diameter - aspect_ratio / 2
            h2 = diameter - aspect_ratio / 2
            overlap_vol = (math.pi * (h1 * h1 / 3 * (3 * radius1 - h1)
                                      + h2 * h2 / 3 * (3 * radius1 - h2)))

            # Calculate dimer's volume by subtracting overlapping volume
            # from sum of volumes of the two spheres
            p_volume = 2 * 4 / 3 * math.pi * radius1**3 - overlap_vol
            print("The particle shape is symmetric dimer.")

        else:
            # Use Monte-Carlo method to find volume of an asymmetric dimer
            p_volume, _, _ = volume_com_moment(spheres, n_points,
                                               n_simulations, mass)

    else:
        # Use Monte-Carlo method to find volume of more complex shapes
        p_volume, _, _ = volume_com_moment(spheres, n_points,
                                           n_simulations, mass)
        print(f"The particle shape is composed of {len(spheres)} spheres.")

    return p_volume


def volume_com_moment(spheres: list,
                      n_points: int = 100000,
                      n_simulations: int = 5,
                      mass: float = 1.0) -> tuple[float, list, list]:
    """Measure the volume, center of mass and moment of inertia of a particle
    shape composed of overlapping spheres using Monte Carlo method.

    The function estimates these properties by randomly sampling points
    and checking if these points fall inside any of the constituent
    spheres of the particle.

    Parameters
    ----------
    spheres: list of tuple
        List of tuples representing the coordinates and radius of each
        constituent sphere.
    n_points: int, optional
        The number of points used in each Monte-Carlo simulation to
        estimate these properties. A higher number improves
        accuracy at the cost of performance.
         (Default value = 100000)
    n_simulations: int, optional
        The number of times a Monte-Carlo simulation is run.
         (Default value = 5)
    mass: float, optional
        The unit mass of a sphere.
         (Default value = 1.0)

    Returns
    -------
    volume : float
        The estimated volume of the particle shape, based on the ratio
        of points inside the spheres to total points, scaled by the
        volume of the bounding box.
    com : list of float
        The estimated center of mass of the particle shape
        (the average position of points inside the spheres).
    inertia : list of float
        The estimated moment of inertia of the particle shape
        (the sum of squared distances from the center of mass for
        all points inside the spheres).
    """

    # Construct a bounding box around the shape and measure its volume
    min_x = min(sphere[0] - sphere[3] for sphere in spheres)
    max_x = max(sphere[0] + sphere[3] for sphere in spheres)
    min_y = min(sphere[1] - sphere[3] for sphere in spheres)
    max_y = max(sphere[1] + sphere[3] for sphere in spheres)
    min_z = min(sphere[2] - sphere[3] for sphere in spheres)
    max_z = max(sphere[2] + sphere[3] for sphere in spheres)
    box_volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)

    # Variables to store the results
    volumes = np.zeros((n_simulations, 1))
    com_s = np.zeros((n_simulations, 3))
    inertia_s = np.zeros((n_simulations, 6))

    # Monte-Carlo simulations
    for i in range(n_simulations):
        # Generate random points within the bounding box
        points = np.random.uniform([min_x, min_y, min_z],
                                   [max_x, max_y, max_z],
                                   size=(n_points, 3))

        # Check if these points are inside any of the spheres
        points_inside = []
        for p in points:
            if any([point_contains(p, sphere) for sphere in spheres]):
                points_inside.append(p)
                com_s[i] += p

        # Volume approximation, ratio of points inside to total points
        # times the box volume
        vol = box_volume * len(points_inside) / n_points
        volumes[i] = vol

        # Center of mass, average of the points weighted by mass
        com_s[i] *= (mass / len(points_inside))

        # Calculate moment of inertia
        for p in points_inside:
            inertia_s[i, 0] += ((com_s[i, 1] - p[1]) ** 2
                                + (com_s[i, 2] - p[2]) ** 2)
            inertia_s[i, 1] += ((com_s[i, 0] - p[0]) ** 2
                                + (com_s[i, 2] - p[2]) ** 2)
            inertia_s[i, 2] += ((com_s[i, 0] - p[0]) ** 2
                                + (com_s[i, 1] - p[1]) ** 2)
            inertia_s[i, 3] -= (com_s[i, 0] - p[0]) * (com_s[i, 1] - p[1])
            inertia_s[i, 4] -= (com_s[i, 0] - p[0]) * (com_s[i, 2] - p[2])
            inertia_s[i, 5] -= (com_s[i, 1] - p[1]) * (com_s[i, 2] - p[2])

        inertia_s[i] *= (mass / len(points_inside))

    # Compute the average values over all simulations
    volume, error = volumes.mean(), volumes.std()
    com = com_s.mean(axis=0)
    inertia = inertia_s.mean(axis=0)

    return volume, com.tolist(), inertia.tolist()


def point_contains(p: np.ndarray, sphere: tuple) -> bool:
    """Check if a point inside a given sphere.

    The Euclidean distance between point and the center of mass of
    the sphere is calculated and checked if it is less than or
    equal to the sphere's radius.

    Parameters
    ----------
    p: np.ndarray
        A 1D array representing the coordinates of the point.
    sphere: tuple
        A tuple representing the coordinates and radius of the sphere
        (x, y, z, radius).

    Returns
    -------
    bool
        True if the point is inside the sphere (including on the surface),
        False otherwise.
    """

    x, y, z, radius = sphere
    distance = np.linalg.norm(p - np.array([x, y, z]))

    return distance <= radius


if __name__ == "__main__":
    # Print the estimated properties of sphere of diameter 1
    volume, com, inertia = volume_com_moment([(0, 0, 0, 0.5)])
    print(f"Volume of sphere is {volume} \n"
          f"Center of mass of sphere is {com} \n"
          f"Moment of inertia of sphere is {inertia}")
