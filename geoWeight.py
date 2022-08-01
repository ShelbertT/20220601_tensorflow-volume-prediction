"""
This script calculates and assigns geographical weighted index to the label.
"""
from globalVar import *
from processBusData import *
import json
from visualization import *
from getConvexHull import get_convex_hull
import pandas as pd
import matplotlib.pyplot as plt
import math

raw_data_folder = get_value('raw_data_folder')
process_data_folder = get_value('process_data_folder')
tensorflow = get_value('tensorflow_folder')
transport_node_bus_folder = get_value('transport_node_bus_folder')


def get_shape_area(polygon):  # polygon[[x, y]] -> float | Get the area of a polygon
    # Calculation is based on [Shoelace Theorem]
    area = 0
    for i in range(len(polygon)):
        current = i
        if current == len(polygon) - 1:
            next_one = 0
        else:
            next_one = i + 1
        area += (polygon[current][0] * polygon[next_one][1]) - (polygon[next_one][0] * polygon[current][1])

    area = abs(area) / 2
    return area


def get_great_circle_distance(lon1, lat1, lon2,
                              lat2):  # Use Haversine formula to calculate great-circle distance in meters
    p = math.pi / 180
    a = 0.5 - math.cos((lat2 - lat1) * p) / 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * (
                1 - math.cos((lon2 - lon1) * p)) / 2
    return 12742 * math.asin(math.sqrt(a)) * 1000  # 2*R*asin...


def get_polygons_in_feature(
        feature_coordinates):  # list[] -> list[list[]] | Extract all polygons in a certain feature and reconstruct them
    all_polygons = []
    # geometries = feature['geometry']['coordinates']

    for i in feature_coordinates:  # This part is the most elegant solution of the entire program. Supported by Luna from Duke-NUS.
        if type(i[0][0]) == float:  # Check if this list is a single polygon
            all_polygons.append(i)  # If so, append it into all_polygons
        else:
            feature_coordinates += i  # If not, append this sub-list into geometries

    return all_polygons


def extract_all_vertices(polygons):  # list[list[list[]]] -> list[] | Extract all vertices from the provided polygons
    vertices = []
    for polygon in polygons:
        for vertex in polygon:
            if vertex not in vertices:
                vertices.append(vertex)

    return vertices


def get_centroid(
        polygons):  # list[list[list[]]] -> list[] | Get the centroid of the polygons. If there is more than one polygon, convex hull will be calculated first.
    if len(polygons) == 1:
        polygon = polygons[0]
    else:
        vertices = extract_all_vertices(polygons)
        polygon = get_convex_hull(vertices)

    area = get_shape_area(polygon)
    centroid_x = 0
    centroid_y = 0

    for i in range(len(polygon)):
        point1 = polygon[i]
        if i == len(polygon) - 1:
            point2 = polygon[0]
        else:
            point2 = polygon[i + 1]

        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]

        centroid_x += (x1 + x2) * (x1 * y2 - x2 * y1)
        centroid_y += (y1 + y2) * (x1 * y2 - x2 * y1)

    centroid_x = abs(centroid_x / (6 * area))
    centroid_y = abs(centroid_y / (6 * area))

    return [centroid_x, centroid_y]


def assign_subzone_centroid_and_population(method=1,
                                           evaluation=False):  # -> .geojson | Assign the centroid of each subzone to its geojson. "method=1" means the pairing method for subzone_names in two files is based on all-uppercase.
    all_subzone = read_json(
        f'{raw_data_folder}/master-plan-2019-subzone-boundary-no-sea/master-plan-2019-subzone-boundary-no-sea.geojson')
    census = pd.read_csv(
        f'{raw_data_folder}/singapore-residents-by-subzone-and-type-of-dwelling-2011-2019/planning-area-subzone-age-group-sex-and-type-of-dwelling-june-2011-2019.csv')

    if method == 1:
        census = census.loc[census['year'] == 2019]
        for index, row in census.iterrows():
            census.loc[index, 'subzone'] = row['subzone'].upper()

    success_count = 0
    total_pop = 0

    for subzone in all_subzone['features']:
        polygons = get_polygons_in_feature(subzone['geometry']['coordinates'])
        centroid = get_centroid(polygons)
        subzone['properties']['Centroid'] = centroid

        if method == 1:
            subzone_name = subzone['properties']['SUBZONE_N']
            population = int(census.loc[census['subzone'] == f'{subzone_name}']['resident_count'].sum())
        else:
            # Rearrange name format
            subzone_name_spl = subzone['properties']['SUBZONE_N'].split()
            subzone_name = ''
            for i in range(len(subzone_name_spl)):
                subzone_name_spl[i] = subzone_name_spl[i].capitalize()
                subzone_name += f'{subzone_name_spl[i]} '
            subzone_name = subzone_name[0:-1]
            population = int(
                census.loc[(census['year'] == 2019) & (census['subzone'] == f'{subzone_name}')]['resident_count'].sum())

        if population != 0:
            success_count += 1
            total_pop += population

        subzone['properties']['Population'] = population

    if evaluation:
        print(f'\nNumber of subzone with population: {success_count}')
        print(f'Total Population: {total_pop}\n')

    write_json(all_subzone, f'{process_data_folder}/3_subzone_with_centroid.json')


def get_polygons(number):  # FOR TESTING PURPOSES ONLY
    all_subzone = read_json(
        f'{raw_data_folder}/master-plan-2019-subzone-boundary-no-sea/master-plan-2019-subzone-boundary-no-sea.geojson')
    polygons = all_subzone['features'][number]['geometry']['coordinates']

    return polygons


def draw_polygon(polygons):  # FOR TESTING PURPOSES ONLY
    cent = get_centroid(polygons)
    print(cent)
    polygon = polygons[0]
    polygon.append(polygon[0])  # repeat the first point to create a 'closed loop'
    xs, ys = zip(*polygon)  # create lists of x and y values

    plt.figure()
    plt.plot(xs, ys)
    plt.scatter(cent[0], cent[1], s=10)
    plt.show()


def get_geographic_weight(distance, function, threshold,
                          band, exclude_self):  # float, str -> float |  Determine the spatial weight of a given distance.
    # Available Functions: Distance_Threshold, Inverse_Distance, Gaussian_Function, Enhanced_Gaussian_Function
    if function == 'Distance_Threshold':
        if distance < threshold:
            weight = 1
        else:
            weight = 0
    elif function == 'Inverse_Distance':
        weight = 1 / distance
    elif function == 'Gaussian_Function':
        weight = math.exp(-pow(distance / band), 2)
    elif function == 'Enhanced_Gaussian_Function':
        if distance < threshold:
            if distance == 0 and exclude_self:
                weight = 0
            else:
                weight = math.exp(-pow(distance / band, 2))
        else:
            weight = 0

    return weight


def calculate_geographic_weight(coords_df, coords_weight,
                                col_name, threshold=2000, band=1000, exclude_self=False):  # DataFrame, list[list[lon, lat, weight]], str -> DataFrame | Calculate the geographic weight for each stop using the weighted point
    total_stops_number = len(coords_df.index)
    print(f'Ready to assign weight, total stops: {total_stops_number}')

    all_assigned_coords = {}  # This dict stores all the coordinates that has already got weight. Since we have duplicate coordinates from different month and daytime, this will save us some time.

    for i in range(total_stops_number):
        stop_lon = coords_df.at[i, 'longitude']
        stop_lat = coords_df.at[i, 'latitude']
        coord_string = f'{stop_lon}-{stop_lat}'
        if coord_string in all_assigned_coords.keys():  # If already exists, no need to calculate again
            weighted_result = all_assigned_coords[coord_string]
        else:
            weighted_result = 0
            for coord_weight in coords_weight:
                distance = get_great_circle_distance(stop_lon, stop_lat, coord_weight[0], coord_weight[1])
                weight = get_geographic_weight(distance, 'Enhanced_Gaussian_Function', threshold=threshold, band=band, exclude_self=exclude_self)
                weighted_result += weight * coord_weight[2]

            # Log the result for this coordinate
            all_assigned_coords[coord_string] = weighted_result

        coords_df.at[i, col_name] = weighted_result  # Add weight to the col name
        # if i % 10000 == 0:
        #     print(f'Progress: {i}/{total_stops_number}')

    return coords_df


def get_population_spatial_index(threshold=200, band=100):
    subzones = read_json(f'{process_data_folder}/3_subzone_with_centroid.json')
    try:
        bus_stops = pd.read_csv(f'{process_data_folder}/4_bus_with_pop_vol.csv')
    except:
        bus_stops = pd.read_csv(f'{process_data_folder}/2_bus_with_coords.csv')

    centroid_with_population = []
    for subzone in subzones['features']:
        weight_point = subzone['properties']['Centroid'].copy()
        weight_point.append(subzone['properties']['Population'])
        centroid_with_population.append(weight_point)

    calculate_geographic_weight(bus_stops, centroid_with_population, f'population_weight_t{threshold}b{band}', threshold=threshold, band=band)

    bus_stops.to_csv(f'{process_data_folder}/4_bus_with_pop_vol.csv', index=False)


def get_stop_spatial_index(threshold=2000, band=1000, exclude_self=False):
    # key_location_number = 1000
    bus_stops = pd.read_csv(f'{process_data_folder}/4_bus_with_pop_vol.csv')
    bus_stops.sort_values(by=['label_in'], ascending=False)
    all_stop_dict = {}

    # First generate a dict to store each stop's coords and total volume
    for i in range(len(bus_stops.index)):
        stop_code = bus_stops.at[i, 'pt_code']
        stop_lon = bus_stops.at[i, 'longitude']
        stop_lat = bus_stops.at[i, 'latitude']
        stop_volume = bus_stops.at[i, 'label_in']

        if stop_code not in all_stop_dict.keys():
            stop_dict = {'lon': stop_lon, 'lat': stop_lat, 'volume': stop_volume}
            all_stop_dict[stop_code] = stop_dict
        elif stop_code in all_stop_dict.keys():
            all_stop_dict[stop_code]['volume'] += stop_volume

    # Structure the data into required format
    all_stop_weight = []
    for key in all_stop_dict.keys():
        stop_info = [all_stop_dict[key]['lon'], all_stop_dict[key]['lat'], all_stop_dict[key]['volume']]
        all_stop_weight.append(stop_info)

    # Invoke the calculate-weight function
    if exclude_self:
        new_col_name = 'surrounding_stop_volume_weight'
    else:
        new_col_name = 'stop_volume_weight'

    calculate_geographic_weight(bus_stops, all_stop_weight, f'{new_col_name}_t{threshold}b{band}', threshold=threshold, band=band, exclude_self=exclude_self)
    bus_stops.to_csv(f'{process_data_folder}/4_bus_with_pop_vol.csv', index=False)


def check_identical_column(col_name):  # Skip the initial settings that have already been run
    try:
        columns = pd.read_csv(f'{process_data_folder}/4_bus_with_pop_vol.csv').columns
    except:
        return False

    if col_name in columns:
        print(f'Duplicate column: {col_name}')
        return True
    return False


def main_geoweight():
    assign_subzone_centroid_and_population(1, True)
    print('FINISHED: assign_subzone_centroid_and_population(1, True)\n')

    # for threshold in [200, 500, 1000, 1500, 2000, 5000, 7000]:
    #     for band in [20, 50, 100, 500, 1000, 2000, 5000]:
    for threshold in [1000, 1500]:
        for band in [500, 1000]:
            duplicate = check_identical_column(f'population_weight_t{threshold}b{band}')
            if not duplicate:
                get_population_spatial_index(threshold=threshold, band=band)
            print(f'Population Weight: Threshold-{threshold}, Band-{band}\n')
    print('FINISHED: get_population_spatial_index()\n')

    # for threshold in [200, 500, 800, 1000, 1500, 2000]:
    #     for band in [25, 50, 75, 100, 500, 1000]:
    for threshold in [1000, 2000]:
        for band in [25, 100]:
            for exclude_self in [False, True]:
                if exclude_self:
                    new_col_name = 'surrounding_stop_volume_weight'
                else:
                    new_col_name = 'stop_volume_weight'
                duplicate = check_identical_column(f'{new_col_name}_t{threshold}b{band}')
                if not duplicate:
                    get_stop_spatial_index(threshold=threshold, band=band, exclude_self=exclude_self)
                print(f'StopVolume Weight: Threshold-{threshold}, Band-{band}, ExcludeSelf-{exclude_self}\n')
    print('FINISHED: get_stop_spatial_index()\n')


if __name__ == "__main__":
    main_geoweight()







