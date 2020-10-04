import sys
import numpy as np


def fixed_point(coordinates_origin=[0, 0, 0]):

    coordinates = coordinates_origin

    return np.array(coordinates)


def strait(direction, velocity, period, coordinates_origin=[0, 0, 0]):
    '''
    coordinates = x, y, z
    velocity[m/s]
    '''

    coordinates = coordinates_origin
    distance = velocity*period
    
    if direction == "x":
        coordinates[0] = coordinates[0] + distance
    if direction == "y":
        coordinates[1] = coordinates[1] + distance
    if direction == "z":
        coordinates[2] = coordinates[2] + distance

    return np.array(coordinates)


def reset_object_coordinates():
    coordinates = np.array([0, 0, 0])
    return coordinates



def calc_distance(coordinates_obj, coordinates_self):

    relative_distance = np.linalg.norm(coordinates_obj - coordinates_self)

    return relative_distance

if __name__ == "__main__":

    period = 10 #sec
    coordinates = np.array([0, 0, 0])

    for time in np.arange(0 ,period,0.1):
        coordinates = strait("x", velocity=10, period=0.1, coordinates_origin=coordinates)
        print("time:" ,round(time,2) , "coordinates:", coordinates)

        coordinates_self = np.array([0, 0, 0])
        distance = calc_distance(coordinates_obj=coordinates, coordinates_self=coordinates_self)
        print(distance)