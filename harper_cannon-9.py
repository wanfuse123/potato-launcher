#!/usr/bin/env python3

#!/usr/bin/env python3

"""
This program simulates the launch of a rocket from a gun barrel. The barrel is broken into segments at a fixed interval for each loop through the calculations. The number of segments 
the barrel is broken into is made in a range from total_barrel_length/rocket_length to 1 in 5 meter intervals. Each pass through the loop is assigned a different color for plotting 
of the curves. 

At the beginning of each segment, there is a shutter that opens and closes so as to shorten the length of the gun as the rocket moves its way up the barrel, decreasing the volume 
behind the rocket and thereby increasing the pressure in the chamber and the forces acting on the rocket. This causes a greater amount of force pushing the rocket forward than 
would normally be on the rocket if it was ignited in the open air. Each shutter closes as the rocket passes past the shutter and moves its way along the defined length of the 
barrel.

The program also takes into account the slope of the gun barrel, which affects the vertical component of gravity acting on the rocket, changing its overall trajectory. Inputs 
to the program include the length and diameter of the barrel, the mass of the rocket, and the specific impulse of the rocket. The program uses numerical integration to calculate 
the velocity of the rocket at various displacements along the barrel and plots this in different colors, one for each segment.

The program also accounts for the distance from the launch site to the center of the Earth and the acceleration due to gravity at the launch site. It should then plot the 
trajectory of the rocket.

Using the Lockeed Martin VentureStar as a starting point for our calculations. The estimated takeoff weight of VentureStar was around 114 metric tons, and it was designed 
to carry up to 20 metric tons of payload to low Earth orbit.

In addition to the existing functionality, we will update the program to include the following:
- The velocities will now be the sum of all the instantaneous velocities in a segment, and the total velocity will be equal to the sum of all the velocities in all the 
  segments that it has passed through, plus any velocity gained from segments it hasn't yet fully passed.
- The program will now output the maximum acceleration and maximum velocity achieved by the rocket during the launch.
- The code is to use the Geodesic class instead of the deprecated Ellipsoid class in geopy.distance module.
"""

# Import required libraries

import math
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic

def rocket_trajectory(rocket_mass, velocity, position, specific_impulse, delta_t, total_barrel_length, rocket_length, diameter, slope):
    """ Calculates the trajectory of a rocket launched from a cannon """

    # Set initial values and parameters
    air_density = 1.2
    Cd = 0.5
    A = 0.25 * np.pi * diameter**2
    barrel_segments = int(total_barrel_length / delta_t)
    shutter_segments = int((rocket_length - total_barrel_length) / delta_t)
    shutter_positions = np.arange(total_barrel_length, rocket_length, 5)
    masses = np.zeros(barrel_segments + shutter_segments + 1)
    masses[0] = rocket_mass
    fuels = np.zeros(barrel_segments + shutter_segments + 1)
    fuels[0] = rocket_mass - masses[0]
    forces = np.zeros(barrel_segments + shutter_segments + 1)
    max_acceleration = 0
    max_velocity = 0
    velocities = np.zeros((barrel_segments + shutter_segments + 1, 3

def initialize_arrays(rocket_mass, velocity, position, total_barrel_length, delta_t):
    """ Initializes arrays for the rocket launch """
    barrel_segments = int(total_barrel_length / delta_t)
    shutter_segments = int((rocket_length - total_barrel_length) / delta_t)
    masses = np.zeros(barrel_segments + shutter_segments + 1)
    masses[0] = rocket_mass
    fuels = np.zeros(barrel_segments + shutter_segments + 1)
    fuels[0] = rocket_mass - masses[0]
    forces = np.zeros(barrel_segments + shutter_segments + 1)
    max_acceleration = 0
    max_velocity = 0
    velocities = np.zeros((barrel_segments + shutter_segments + 1, 3))
    positions = np.zeros((barrel_segments + shutter_segments + 1, 3))
    positions[0] = position
    velocities[0] = velocity
    accelerations = np.zeros((barrel_segments + shutter_segments + 1, 3))
    return masses, fuels, forces, max_acceleration, max_velocity, velocities, positions, accelerations


def launch_rocket(positions, velocities, accelerations, masses, fuels, forces, max_acceleration, max_velocity, specific_impulse, air_density, Cd, A, total_barrel_length, delta_t, rocket_length, slope):
""" Launches a rocket from a cannon and returns the final values of various parameters """
    # Calculate the initial values for the rocket's motion
masses[0] = rocket_mass
fuels[0] = rocket_mass - masses[0]
forces[0] = barrel_force(0, total_barrel_length, slope, diameter, Cd, A, air_density, velocities[0], masses[0], fuels[0], specific_impulse)
acceleration = forces[0] / masses[0]
if abs(acceleration[0]) > abs(max_acceleration):
    max_acceleration = acceleration[0]
if abs(velocities[0][0]) > abs(max_velocity):
    max_velocity = velocities[0][0]
accelerations[0] = acceleration
velocity = runge_kutta(lambda v, t: acceleration[:2], velocities[0], 0, delta_t)
position = runge_kutta(lambda p, t: velocity, positions[0], 0, delta_t)
velocities[0] = velocity
positions[0] = position

# Launch the rocket from the barrel
time = delta_t
while position[0] < total_barrel_length:
    masses[0] = rocket_mass
    fuels[0] = rocket_mass - masses[0]
    forces[0] = barrel_force(time, total_barrel_length, slope, diameter, Cd, A, air_density, velocities[0], masses[0], fuels[0], specific_impulse)
    acceleration = forces[0] / masses[0]
    if abs(acceleration[0]) > abs(max_acceleration):
        max_acceleration = acceleration[0]
    if abs(velocities[0][0]) > abs(max_velocity):
        max_velocity = velocities[0][0]
    accelerations[0] = acceleration
    velocity = runge_kutta(lambda v, t: acceleration[:2], velocities[0], time, delta_t)
    position = runge_kutta(lambda p, t: velocity, positions[0], time, delta_t)
    velocities[0] = velocity
    positions[0] = position
    time += delta_t

# Launch the rocket from the shutter
mdot = 3575.719938710052
max_fuel = 20000
fuel = 0
shutter_segments = int((total_barrel_length - rocket_length) / 5)
shutter_positions = np.zeros(shutter_segments)
for i in range(shutter_segments):
    shutter_positions[i] = rocket_length + 5 * i

for i in range(shutter_segments):
    shutter_position = shutter_positions[i]
    if shutter_position > rocket_length:
        break
    shutter_area = math.pi / 4 * diameter**2 * shutter_position
    while fuel < max_fuel and shutter_position > 0:
        masses[i+1] = rocket_mass - fuel
        fuels[i+1] = fuel
        forces[i+1] = barrel_force(time, total_barrel_length, slope, diameter, Cd, A, air_density, velocities[i], masses[i+1], fuels[i+1], specific_impulse)
        acceleration = forces[i+1] / masses[i+1]
        if abs(acceleration[0]) > abs(max_acceleration):
            max_acceleration = acceleration[0]
        if abs(

  # Launch the rocket from the shutter
    mdot = 3575.719938710052
    max_fuel = 20000
    fuel = 0
    shutter_segments = int((total_barrel_length - rocket_length) / 5)
    shutter_positions = np.zeros(shutter_segments)
    for i in range(shutter_segments):
        shutter_positions[i] = (i + 1) * 5 + rocket_length
    for i in range(shutter_segments):
        shutter_position = shutter_positions[i]
        if shutter_position > rocket_length:
            break
        shutter_area = math.pi / 4 * diameter**2 * shutter_position
        while fuel < max_fuel and shutter_position > 0:
            masses[i+1] = rocket_mass - fuel
            fuels[i+1] = fuel
            forces[i+1] = barrel_force(time, total_barrel_length, slope, diameter, Cd, A, air_density, velocities[i], masses[i+1], fuels[i+1], specific_impulse)
            acceleration = forces[i+1] / masses[i+1]
            if abs(acceleration[0]) > abs(max_acceleration):
                max_acceleration = acceleration[0]
            if abs(velocities[i][0]) > abs(max_velocity):
                max_velocity = velocities[i][0]
            accelerations[i+1] = acceleration
            velocity = runge_kutta(lambda v, t: acceleration[:2], velocities[i], time, delta_t)
            position = runge_kutta(lambda p, t: velocity, positions[i], time, delta_t)
            velocities[i+1] = velocity
            positions[i+1] = position
            time += delta_t
            shutter_position -= delta_t * velocity[0]
            if shutter_position < rocket_length:
                fuel += mdot * delta_t
                if fuel > max_fuel:
                    fuel = max_fuel
    return positions, velocities, accelerations, masses, fuels, forces, max_acceleration, max_velocity


 
def main():
    # Define the input parameters
    total_barrel_length = 10000  # m
    rocket_length = 80  # m
    diameter = 6.5  # m
    rocket_mass = 114000  # kg
    specific_impulse = 311  # s
    slope = math.radians(5)  # degrees

    # Set the initial velocity and position
    distance = 250000  # m
    lat1 = math.radians(28.5)  # degrees
    lon1 = math.radians(-80.5)  # degrees
    lat2 = math.asin(math.sin(lat1) * math.cos(distance / 6371000) + math.cos(lat1) * math.sin(distance / 6371000) * math.cos(0))
    lon2 = lon1 + math.atan2(math.sin(0) * math.sin(distance / 6371000) * math.cos(lat1), math.cos(distance / 6371000) - math.sin(lat1) * math.sin(lat2))
    position = np.array([geodesic((lat1, lon1), (lat2, lon2)).m, 0, 0])
    velocity = np.array([2000, 7000, 0])  # m/s

    # Set the time step and run the simulation
    delta_t = 0.1  # s
    times, positions, velocities, accelerations, masses, fuels, forces, max_acceleration, max_velocity = rocket_trajectory(rocket_mass, velocity, position, specific_impulse, delta_t, total_barrel_length, rocket_length, diameter, slope)

    # Print the results
    print(f"Altitude change: {positions[-1, 1] - positions[0, 1]:.2f} m")
    print(f"Delta-v: {np.linalg.norm(velocities[-1] - velocities[0]):.2f} m/s")
    print(f"Specific impulse change: {specific_impulse * math.log(rocket_mass / masses[-1]):.2f} s")
    print(f"Fuel usage: {max_fuel - fuels[-1]:.2f} kg")
    print(f"Maximum acceleration: {max_acceleration:.2f} m/s^2")
    print(f"Maximum velocity: {max_velocity:.2f} m/s")


if __name__ == '__main__':
    main()

