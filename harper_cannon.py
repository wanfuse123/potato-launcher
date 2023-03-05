#!/usr/bin/env python3

# The program simulates the launch of a rocket from a gun barrel, which is broken into segments at a fixed interval for each loop through the calculations. The number of segments the 
# barrel is broken is made in a range from total_barrel_length/rocket_length to 1 in 5 meter intervals. Each pass through the loop is assigned a different color for plotting of the  
# curves. At the beginning of each segment, is a shutter that opens and closes so as to shorten the length of the gun as the rocket moves its way up the barrel. Theirby decreasing the
# volume behind the rocket and theirby increasing the pressure in the chamber and the forces acting on the rocket causing a great amount of force pushing the rocket forward than would 
# normally be on the rocket if it was ignited in the open air. Each shutter closes as the rocket passes past the shutter and moves its way along the defined length of the barrel. The
# program also takes into account the slope of the gun barrel, which affects the vertical component of grivity acting on the rocket (changing its overall trajectory. Inputs to the 
# program include the length and diameter of the barrel, the mass of the rocket, the specific impulse of the rocket. The program uses numerical integration to calculate the velocity of 
# the rocket at various displacements along the barrel, and plots this in different colors  one for each in a range of total number of segments. The program also accounts for the 
# distance from the launch site to the center of the Earth and the acceleration due to gravity at the launch site. It should then plot the trajectory of the rocket. 

# NOTE: It looks like the geopy.distance module has changed and the Ellipsoid class is no longer available. 
# Instead, you can use the Geodesic class to compute the distance between two points on the Earth's surface.

# using Lockeed Martin VentureStar as a starting point for our calculations. The estimated takeoff weight of VentureStar was around 114 metric tons, and it was 
# designed to carry up to 20 metric tons of payload to low Earth orbit.

# Import required libraries

import math
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import geopy 
import pandas 

# Constants
G = 6.6743e-11 # gravitational constant in m^3/(kg*s^2)
M = 5.97e24 # mass of the Earth in kg
specific_impulse = 311 # specific impulse of the rocket in seconds (for Saturn V rocket)
exit_area = math.pi * (0.67 / 2)**2 # area of the nozzle exit in square meters (for Saturn V rocket)
exit_pressure = 101325 # pressure of the exhaust gases in pascals
default_slope = 0 # default slope of the barrel in degrees

# Inputs for Saturn V rocket
# rocket_length = 110.6 # length of the rocket in meters
# rocket_diameter = 10.1 # diameter of the rocket in meters
# rocket_mass_full = 2970000 # mass of the rocket when fully fueled in kg
# rocket_mass_empty = 131000 # mass of the rocket when empty in kg

# Inputs for Lockeed Martin VentureStar rocket
rocket_length = 52.4 # length of the rocket in meters
rocket_diameter = 8.38 # diameter of the rocket in meters
rocket_mass_full = 114000 # mass of the rocket when fully fueled in kg
rocket_mass_empty = 28000 # mass of the rocket when empty in kg

# Inputs for gun
barrel_length = 3000 # length of the barrel in meters
barrel_diameter = 1 # diameter of the barrel in meters
total_length_of_rocket = rocket_length # total length of the rocket in meters

# Get latitude and altitude of launch site (e.g. near the equator)
launch_site = (0.0236, -78.5249) # latitude and longitude of launch site (coordinates of the Chimborazo mountain in Ecuador)
altitude = 8267 # altitude of launch site in meters (altitude of Chimborazo mountain) + 2km starting point
geod = geopy.distance.Geodesic(a=6378137, f=1/298.257223563)

distance_to_center_of_earth = 6371000 + altitude # distance to center of Earth in meters (approximate mean Earth radius is 6371 km)

r = distance_to_center_of_earth
latitude = launch_site[0]
coslat = math.cos(math.radians(latitude))
sinlat = math.sin(math.radians(latitude))

# standard acceleration due to gravity in m/s^2
g0 = 9.780327 * (1 + 0.0053024 * sinlat**2 - 0.0000058 * math.sin(2 * latitude)**2) - 0.000003086 * altitude 

# Initializations
rocket_position = 0 # initial position of the rocket in meters
interval_lengths = []
rocket_position = 0
#shutter_positions = [x * rocket_length / 10 for x in range(11)]
#shutter_positions = [x * total_length_of_rocket / 10 for x in range(11)]
#interval_frequency = shutter_positions[1] - shutter_positions[0]
# Generate the shutter positions based on the number of divisions and min/max spacing
num_divisions = 10  # number of shutter positions to generate
min_spacing = 2 * rocket_length  # minimum spacing between shutter positions
max_spacing = barrel_length  # maximum spacing between shutter positions
divisions = [i / (num_divisions - 1) for i in range(num_divisions)]  # create a list of fractional divisions
divisions = [min_spacing + (max_spacing - min_spacing) * d for d in divisions]  # scale divisions to be within min/max spacing range
shutter_positions = [sum(divisions[:i]) for i in range(1, num_divisions + 1)]  # generate shutter positions as cumulative sum of divisions

# Calculate the segment lengths based on the distance to the next shutter
interval_lengths = []
rocket_position = 0
accelerations = []  # Add this line to initialize the list

for i in range(len(shutter_positions) - 1):
    distance_to_next_shutter = shutter_positions[i+1] - rocket_position
    while distance_to_next_shutter > 0:
        segment_length = min(distance_to_next_shutter, max_spacing)
        interval_lengths.append(segment_length)
        distance_to_next_shutter -= segment_length
        rocket_position += segment_length

num_intervals = len(interval_lengths) # number of intervals used in the simulation

rocket_velocity = 0
rocket_mass = rocket_mass_full
shutter_area = 0
velocities = []
altitudes = [distance_to_center_of_earth]  # list to store altitudes of the rocket


for i in range(num_intervals):
    segment_length = interval_lengths[i]

    segment_radius = barrel_diameter / 2 * (1 - (rocket_position - segment_length / 2) / barrel_length)
    segment_area = math.pi * segment_radius ** 2
    mdot = rocket_mass * g0 / specific_impulse
    ve = specific_impulse * g0
    thrust = mdot * ve + exit_area * (exit_pressure - shutter_area / exit_area * exit_pressure)

    if rocket_position + segment_length / 2 >= distance_to_center_of_earth:
        break

    force_gravity = G * M * rocket_mass / r**2
    force_net = thrust - force_gravity
    acceleration = force_net / rocket_mass
    rocket_velocity += acceleration * segment_length
    velocities.append(rocket_velocity)

    # Compute altitude change for current segment
    print("specific_impulse:", specific_impulse)
    print("g0", g0)
    print("rocket_mass:", rocket_mass)
    print("mdot:", mdot)
    print("segment_length:", segment_length)
    altitude_change = (specific_impulse * g0 * math.log(rocket_mass / (rocket_mass - mdot * segment_length)) - g0 * segment_length) / 2
    altitudes.append(altitudes[-1] + altitude_change)

    # Update rocket mass
    min_mass = rocket_mass_empty * 0.05  # set a minimum mass cutoff value (5% of the empty mass)
    rocket_mass = max(rocket_mass - mdot * segment_length, min_mass)
    if rocket_mass <= min_mass:
        print("Rocket has reached minimum mass.")
    data = {'Variable Name': ['rocket_position', 'segment_length', 'segment_radius', 'segment_area', 'mdot', 've', 'thrust', 'force_gravity', 'force_net', 'acceleration', 'rocket_velocity', 'altitude_change', 'rocket_mass', 'min_mass'],
            'Value': [rocket_position, segment_length, segment_radius, segment_area, mdot, ve, thrust, force_gravity, force_net, acceleration, rocket_velocity, altitude_change, rocket_mass, min_mass]}
    table = pd.DataFrame(data)
    print(table)

    shutter_area += segment_area

    # Update rocket position based on velocity and acceleration
    delta_t = segment_length / rocket_velocity
    rocket_position += rocket_velocity * delta_t + 0.5 * acceleration * delta_t ** 2
    rocket_velocity += acceleration * delta_t



# Plotting
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
for i in range(1, len(shutter_positions)):
    start_index = sum([len(interval_lengths[:j]) for j in range(i)])
    end_index = sum([len(interval_lengths[:j]) for j in range(i+1)])
    plt.plot(interval_lengths[start_index:end_index], velocities[start_index:end_index], color=colors[i%7], linestyle=linestyles[i%4], label='Velocity Interval {}'.format(i))
    plt.plot(interval_lengths[start_index:end_index], accelerations[start_index:end_index], color=colors[i%7], linestyle=linestyles[i%4], label='Acceleration Interval {}'.format(i))
    plt.plot(interval_lengths[start_index:end_index], altitudes[start_index:end_index], color=colors[i%7], linestyle=linestyles[i%4], label='Altitude Interval {}'.format(i))
