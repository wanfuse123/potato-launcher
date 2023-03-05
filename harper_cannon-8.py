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

def rocket_trajectory(mass, velocity, position, specific_impulse, delta_t, total_barrel_length, rocket_length, diameter, slope):
    # Define the constants
    G = 6.6743e-11
    M = 5.972e24
    g0 = 9.81

    # Define the gravitational force and the rocket engine force functions
    def gravity_force(m):
        norm = np.linalg.norm(position[:2])
        return np.array([-G * M * m / norm**3 * position[0], -G * M * m / norm**3 * position[1], 0])

    def engine_force(m, time, shutter_position):
        volume = math.pi / 4 * diameter**2 * shutter_position
        chamber_pressure = (mass + fuel) / volume
        if shutter_position > rocket_length:
            thrust = specific_impulse * g0 * mdot
        else:
            thrust = 0
        return np.array([thrust, -chamber_pressure * math.pi / 4 * diameter**2, 0])

    # Define the Runge-Kutta function for the rocket's motion
    def runge_kutta(f, y0, x0, h):
        """ Fourth order Runge-Kutta method for solving differential equations """
        x = x0
        y = y0.copy()
        while True:
            yield y
            k1 = f(y, x) * h
            k2 = f(y + 0.5 * k1, x + 0.5 * h) * h
            k3 = f(y + 0.5 * k2, x + 0.5 * h) * h
            k4 = f(y + k3, x + h) * h
            y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            x += h
            if x >= 1.0:
                break
    def rocket_trajectory(rocket_mass, velocity, position, specific_impulse, delta_t, total_barrel_length, rocket_length, diameter, slope):
    """ Calculates the trajectory of a rocket launched from a cannon """
    air_density = 1.2
    Cd = 0.5
    A = 0.25 * pi * diameter**2
    barrel_segments = int(total_barrel_length / delta_t)
    shutter_segments = int((rocket_length - total_barrel_length) / delta_t)
    time = 0
    dt = delta_t
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
    while time < total_barrel_length:
        masses[0] = rocket_mass
        fuels[0] = rocket_mass - masses[0]
        forces[0] = barrel_force(time, total_barrel_length, slope, diameter, Cd, A, air_density, velocities[0], masses[0], fuels[0], specific_impulse)
        acceleration = forces[0] / masses[0]
        if abs(acceleration[0]) > abs(max_acceleration):
            max_acceleration = acceleration[0]
        if abs(velocities[0][0]) > abs(max_velocity):
            max_velocity = velocities[0][0]
        accelerations[0] = acceleration
        #position = runge_kutta(lambda p, t: velocity[:2], position, time, dt)
        #velocity = runge_kutta(lambda v, t: acceleration[:2], velocity, time, dt)
        velocity = runge_kutta(lambda v, t: acceleration[:2], velocity, time, dt)
        position = runge_kutta(lambda p, t: velocity, position, time, dt)
        velocity = runge_kutta(lambda v, t: acceleration, velocity, time, dt)
        position = runge_kutta(lambda p, t: velocity[:2], position, time, dt)

        positions[0] = position
        
        velocities[0] = velocity
        time += dt
    for i in range(shutter_segments):
    
    # Set the initial values for the rocket's motion
    shutter_segments = int((total_barrel_length - rocket_length) / 5)
    segment_length = (total_barrel_length - rocket_length) / shutter_segments
    end_time = total_barrel_length / velocity[0]
    mdot = 3575.719938710052
    max_fuel = 20000
    fuel = 0
    times = np.arange(0, end_time, delta_t)
    print("length of times: ", len(times))

    # Fix initial values of arrays to have three columns instead of two
    positions = np.zeros((len(times), 3)) 
    velocities = np.zeros((len(times), 3))
    accelerations = np.zeros((len(times), 3))

    masses = np.zeros(len(times))
    fuels = np.zeros(len(times))
    forces = np.zeros((len(times), 3))  # update forces to have three components
    shutter_positions = np.zeros(shutter_segments)
    for i in range(shutter_segments):
        shutter_positions[i] = total_barrel_length - (total_barrel_length - rocket_length) * (i + 1) / shutter_segments
    positions[0, :2] = position[:2]
    positions[0, 2] = position[2]
    velocities[0] = np.array([velocity[0], velocity[1], velocity[2]])  # update velocities
    accelerations[0] = gravity_force(mass) / mass  # remove error fix
    masses[0] = mass
    fuels[0] = fuel

    # Integrate the rocket's motion using the Runge-Kutta method
    segment_index = 0
    for i in range(1, len(times)):
        time = times[i]
        dt = times[i] - times[i-1]
        position = positions[i-1]
        velocity = velocities[i-1]
        mass = masses[i-1]
        fuel = fuels[i-1]
        acceleration = gravity_force(mass) / mass
        if position[0] > shutter_positions[segment_index]:
            acceleration += engine_force(mass, time, shutter_positions[segment_index])
            segment_index += 1
        # velocity = runge_kutta(lambda v, t: acceleration, velocity, time, dt)   #removed error fix
        # position = runge_kutta(lambda p, t: velocity, position, time, dt)       #removed error fix    
        velocity = runge_kutta(lambda v, t: acceleration, velocity, time, dt)
        position = runge_kutta(lambda p, t: velocity[:2], position, time, dt)
        mass = mass - mdot * dt
        fuel = min(fuel + mdot * dt, max_fuel)
        forces[i] = gravity_force(mass) + engine_force(mass, time, shutter_positions[segment_index-1])
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

    # Calculate the maximum acceleration and maximum velocity achieved by the rocket during launch
    max_acceleration = max(np.linalg.norm(accelerations, axis=1))
    max_velocity = max(np.linalg.norm(velocities, axis=1))

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
    print(f"Maximum acceleration: {max_acceleration:.2f} m/s^2")
    print(f"Maximum velocity: {max_velocity:.2f} m/s")

    plt.show()

   
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

