"""
This script provides the functions for calculating the convex hull.
"""
import math
from decimal import Decimal


def generate_vector(start, end):
    vector = [float(Decimal(f'{end[0]}')-Decimal(f'{start[0]}')), float(Decimal(f'{end[1]}')-Decimal(f'{start[1]}'))]
    return vector


# list[x, y], list[x, y] -> cos of angle | get the cos of the angle between two vectors
def get_vector_angle(vector1, vector2):
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    length1 = math.sqrt(vector1[0] * vector1[0] + vector1[1] * vector1[1])
    length2 = math.sqrt(vector2[0] * vector2[0] + vector2[1] * vector2[1])
    angle_cos = dot_product / (length1 * length2)
    return angle_cos


# point[] -> bool | Check whether middle point is on the left of the vector[start, end]
def whether_left(start, middle, end):
    vector1 = generate_vector(start, middle)
    vector2 = generate_vector(middle, end)
    cross_product_value = Decimal(f'{vector1[0]}') * Decimal(f'{vector2[1]}') - Decimal(f'{vector1[1]}') * Decimal(f'{vector2[0]}')

    if cross_product_value > 0:
        return False
    elif cross_product_value < 0:
        return True


# vertices[vertex[x, y]] -> polygon[vertex[x, y]] | Generate the convex hull of input vertices
def get_convex_hull(vertices):
    # Set the point with the smallest lat as the origin.
    smallest_lat = 100
    origin_index = 0
    for i in range(len(vertices)):
        if vertices[i][1] < smallest_lat:
            smallest_lat = vertices[i][1]
            origin_index = i
    origin = vertices.pop(origin_index)

    # Consider the remaining n-1 points and sort them by polar angle in counterclockwise order around origin
    polar_sequence = []
    angle_index = {}
    x_axis = [1, 0]

    for i in range(len(vertices)):
        vector = generate_vector(origin, vertices[i])
        angle = get_vector_angle(x_axis, vector)
        angle_index[angle] = i
    sequence = list(angle_index.keys())
    sequence.sort(reverse=True)

    for angle in sequence:
        polar_sequence.append(vertices[angle_index[angle]])  # The first point is the one with the smallest polar angle

    # Traverse all the remaining points to generate convex hull
    convex_hull = [origin, polar_sequence.pop(0)]

    while len(polar_sequence) > 0:
        left = whether_left(convex_hull[-2], convex_hull[-1], polar_sequence[0])
        if left:
            convex_hull.pop(-1)
        else:
            convex_hull.append(polar_sequence.pop(0))

    return convex_hull

#
# if __name__ == '__main__':
#     main()
