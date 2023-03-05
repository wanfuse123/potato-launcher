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
import numpy as np
import matplotlib.pyplot as plt

def rocket_trajectory(mass, velocity, position, specific_impulse, delta_t, total_barrel_length, rocket_length, diameter, slope):
    # Define the constants
    G = 6.6743e-11
    M = 5.972e24
    g0 = 9.81

    # Define the gravitational force and the rocket engine force functions
    def gravity_force(m):
        return -G * M * m / np.linalg.norm(position)**3 * position

    def engine_force(m, time):
        shutter_position = total_barrel_length - (total_barrel_length - rocket_length) * time / end_time
        volume = math.pi / 4 * diameter**2 * shutter_position
        chamber_pressure = (mass + fuel) / volume
        if shutter_position > rocket_length:
            thrust = specific_impulse * g0 * mdot
        else:
            thrust = 0
        return np.array([thrust, -chamber_pressure * math.pi / 4 * diameter**2])

    # Define the Runge-Kutta function for the rocket's motion
    def runge_kutta(f, y, x, h):
        k1 = f(y, x) * h
        k2 = f(y + 0.5 * k1, x + 0.5 * h) * h
        k3 = f(y + 0.5 * k2, x + 0.5 * h) * h
        k4 = f(y + k3, x + h) * h
        return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # Set the initial values for the rocket's motion
    end_time = total_barrel_length / velocity[0]
    mdot = 3575.719938710052
    max_fuel = 20000
    fuel = 0
    times = np.arange(0, end_time, delta_t)
    positions = np.zeros((len(times), 2))
    velocities = np.zeros((len(times), 2))
    accelerations = np.zeros((len(times), 2))
    masses = np.zeros(len(times))
    fuels = np.zeros(len(times))
    forces = np.zeros((len(times), 2))
    positions[0] = position
    velocities[0] = velocity
    accelerations[0] = (gravity_force(mass) + engine_force(mass, 0)) / mass
    masses[0] = mass

    # Integrate the rocket's motion using the Runge-Kutta method
    for i in range(1, len(times)):
        time = times[i]
        dt = times[i] - times[i-1]
        position = positions[i-1]
        velocity = velocities[i-1]
        mass = masses[i-1]
        fuel = fuels[i-1]
        shutter_position = total_barrel_length - (total_barrel_length - rocket_length) * time / end_time
        acceleration = (gravity_force(mass) + engine_force(mass, time) + np.array([0, -mass * g0 * math.sin(slope)])) / mass
        velocity = runge_kutta(lambda v, t: acceleration, velocity, time, dt)
        position = runge_kutta(lambda p, t: velocity, position, time, dt)
        mass -= mdot * dt
        fuel = min(fuel + mdot * dt, max_fuel)
        forces[i] = gravity_force(mass) + engine_force(mass, time) + np.array([0, -mass * g0 * math.sin(slope)])
        accelerations[i] = forces[i] / mass
        positions[i] = position
        velocities[i] = velocity
        masses[i] = mass
        fuels[i] = fuel

    # Calculate the altitude change
    delta_altitude = positions[-1, 1] - positions[0, 1]

    # Calculate the delta-v
    delta_v = np.linalg.norm(velocities[-1] - velocities[0])

    # Calculate the specific impulse change
    delta_specific_impulse = specific_impulse * math.log(mass / masses[-1])

    # Calculate the fuel usage
    fuel_usage = max_fuel - fuels[-1]

    # Plot the rocket's trajectory
    colors = plt.cm.rainbow(np.linspace(0, 1, len(times)))
    plt.figure()
    plt.axis('equal')
    for i in range(len(times)):
        plt.plot(positions[i, 0], positions[i, 1], '.', color=colors[i])
    plt.xlabel('Distance (m)')
    plt.ylabel('Altitude (m)')
    plt.title('Rocket Trajectory')

    # Print the results
    print(f"Altitude change: {delta_altitude:.2f} m")
    print(f"Delta-v: {delta_v:.2f} m/s")
    print(f"Specific impulse change: {delta_specific_impulse:.2f} s")
    print(f"Fuel usage: {fuel_usage:.2f} kg")

    plt.show()


