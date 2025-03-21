"""
Module for importing particle shape and simulation data and preparing
them for analysis.
"""

import pandas as pd


class DataPreprocessing:
    """
    Import simulation data and prepare them for analysis.

    The simulation file should contain information
    per constituent sphere in the first six columns: 'id', 'x', 'y',
    'z', 'radius', 'Voronoi_volume'. This information is used to
    calculate geometric centroids and Voronoi volume per particle.

    Parameters
    ----------
    simulation_filepath : str
        Path to the file containing simulation data.
    box_width : float
        Width of the bulk region (simulation box width). This is scaled
        by the diameter of the first sphere in the particle,
        which is set to 1 in the simulation.
    box_length : float
        Length of the bulk region (simulation box length). This is also
        scaled by the diameter of the first sphere in the particle.

    Attributes
    ----------
    simulation_filepath : str
        Path to the file containing simulation data.
    sphere_data : pd.DataFrame
        A Dataframe containing the following columns: 'id', 'x', 'y',
        'z', 'radius', 'Voronoi_volume', and engineered columns
        'z_lower', 'z_upper'.
    particle_data : pd.DataFrame
        A Dataframe containing the following columns: 'id',
        'Voronoi_volume', 'z_lower', 'z_upper', 'centroid_x',
        'centroid_y', 'centroid_z', engineered from sphere_data.
    """

    def __init__(self, simulation_filepath: str, box_width: float,
                 box_length: float) -> None:
        """Initialize the DataPreprocessing class with simulation data.

        This constructor reads simulation file and
        processes the data to extract information per particle
        in the simulation.

        Parameters
        ----------
        simulation_filepath : str
            Path to the file containing simulation data.
        box_width : float
            Width of the bulk region (simulation box width). This is scaled
            by the diameter of the first sphere in the particle,
            which is set to 1 in the simulation.
        box_length : float
            Length of the bulk region (simulation box length). This is also
            scaled by the diameter of the first sphere in the particle.
        """

        self.simulation_filepath = simulation_filepath
        self.box_width = box_width
        self.box_length = box_length
        self.sphere_data = self.read_simulation_data()
        self.particle_data = self.extract_particle_data()

    def read_simulation_data(self):
        """Read the simulation data from a .csv file the given location
         into DataFrame.

        The simulation data is expected to include the information
        ('id', 'x', 'y', 'z', 'radius', 'Voronoi_volume') per sphere in
        the first six columns, respectively.

        Returns
        -------
        df : pd.DataFrame, None
            A dataframe of the simulation data with the columns: 'id',
            'x', 'y', 'z', 'radius', 'Voronoi_volume'.
        """

        # List of column names to be used in the dataframe
        cols_list = ['id', 'x', 'y', 'z', 'radius', 'Voronoi_volume']

        # Read .csv file with pandas
        try:
            df = pd.read_csv(self.simulation_filepath,
                             header=None,
                             usecols=[0, 1, 2, 3, 4, 5],
                             names=cols_list)
            return df
        except FileNotFoundError:
            print(f"Error: The file at {self.simulation_filepath} was not"
                  f" found.")
        except pd.errors.EmptyDataError:
            print("Error: The file is empty.")
        except pd.errors.ParserError:
            print("Error: There was a problem parsing the file.")

    def extract_particle_data(self):
        """Extract information per particle from sphere_data.

        Voronoi volume of a particle is obtained by summing Voronoi
        volumes of its constituent spheres. Geometric centroid of
        a particle is obtained by averaging the coordinates of its
        constituent spheres.

        Returns
        -------
        particle_data : pd.DataFrame
            A DataFrame containing the following columns: 'id',
            'Voronoi_volume', 'z_lower', 'z_upper', 'centroid_x',
            'centroid_y', 'centroid_z', engineered from sphere_data.
        """

        # Find the lowest point of each sphere in the z-direction
        self.sphere_data["z_lower"] = (
            self.sphere_data["z"] - self.sphere_data["radius"]
        )

        # Find the highest point of each sphere in the z-direction
        self.sphere_data["z_upper"] = (
            self.sphere_data["z"] + self.sphere_data["radius"]
        )

        # Check if any constituent spheres found in the opposite side of
        # the box in lateral directions(x-y), e.g. one is at x ~ 0 and
        # the other is at x ~ box_width
        self.sphere_data['x_diff'] = self.sphere_data.groupby('id')[
            'x'].transform(lambda x: x - x.min())

        self.sphere_data['y_diff'] = self.sphere_data.groupby('id')[
            'y'].transform(lambda y: y - y.min())

        # Determine this criteria
        periodic_x = self.box_width - 2 * self.sphere_data['radius'].max()
        mask_x = self.sphere_data['x_diff'] > periodic_x

        periodic_y = self.box_width - 2 * self.sphere_data['radius'].max()
        mask_y = self.sphere_data['y_diff'] > periodic_y

        # Save original x and y values
        self.sphere_data['x_original'] = self.sphere_data['x']
        self.sphere_data['y_original'] = self.sphere_data['y']

        # Move any constituent spheres in the opposite side of the box
        # to the other side (due to the periodicity)

        self.sphere_data.loc[mask_x, 'x'] = (self.sphere_data.loc[
                                             mask_x, 'x'] - self.box_width)

        self.sphere_data.loc[mask_y, 'y'] = (self.sphere_data.loc[
                                             mask_y, 'y'] - self.box_width)

        # Find Voronoi volume, and the lowest and highest points of
        # each particle in the z-direction
        particle_data = self.sphere_data.groupby("id").agg(
            dict(Voronoi_volume="sum", z_lower="min", z_upper="max"))

        # Find geometric centroid of each particle
        particle_data[["x", "y", "z"]] = (
            self.sphere_data.groupby("id")[["x", "y", "z"]]
            .mean()
        )

        particle_data = particle_data.reset_index()

        return particle_data
