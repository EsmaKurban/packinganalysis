"""
Example usage of the package `packinganalysis` for analysis of sphere
packings.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from packinganalysis import read_shape_data, particle_volume
from packinganalysis import PackingAnalysis

# Paths for sphere shape file and simulation file of sphere packing
shape_filepath = "../src/packinganalysis/data/sphere/shape.csv"
simulation_filepath = "../src/packinganalysis/data/sphere/dump.csv"

width = 20      # Simulation box width scaled by sphere diameter d
length = 20     # Simulation box length scaled by sphere diameter d

# Read shape file and calculate volume of the shape
spheres = read_shape_data(shape_filepath)
p_vol = particle_volume(spheres)

# Initialize the class PackingAnalysis
sphere = (PackingAnalysis(spheres, p_vol, simulation_filepath,
                          box_width=width,
                          box_length=length))

print(f"Sphere volume is {sphere.p_vol}.")

print(f"Packing density measured for sphere packing with Voronoi method is "
      f"{sphere.voronoi_density()} and with centroid method is "
      f"{sphere.centroid_density(box_height=5)}.")

print(f"Average contact and coordination number of sphere packing are "
      f"{sphere.contact_analysis()}.")

# Plot radial distribution function
rdf = sphere.radial_distribution_function(shell_width=0.025)
plt.figure(figsize=(7, 5))
plt.plot(rdf[:, 0], rdf[:, 1], color='red', linestyle='solid',
         linewidth=2, label='All pairs')
plt.ylabel(r'g(r)', size=14)
plt.xlabel(r'r/d', size=14)
plt.legend()
plt.tick_params(direction='in')
plt.show()

# Calculate order parameters for sphere packing
sphere.bond_orientational_order_parameters()
sphere.particle_data.to_csv(
    "../src/packinganalysis/data/sphere/sphere_analysis.csv")

# Plot local order parameters q4 and q6 distribution of sphere packing
fig, ax = plt.subplots(1, 1, figsize=(5.6, 4.2))
sns.scatterplot(x='q4_tilda_bar', y='q6_tilda_bar', data=sphere.particle_data)
plt.xlabel(r'$\overline{\widetilde{q}}_{4}$', size=16, labelpad=5)
plt.ylabel(r'$\overline{\widetilde{q}}_{6}$', size=16, labelpad=5)
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.yticks([0, 0.25, 0.5, 0.75, 1])
plt.xlim(-0.1, 1)
plt.ylim(0, 1.025)
plt.tight_layout()
plt.show()
