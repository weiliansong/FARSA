from collections import OrderedDict

img_means = [123.68, 116.78, 103.94]

spec = OrderedDict()

spec['area_type'] = {
    'num_classes'   : 2,
    'needs_binning' : False,
    'csv_index'     : 'Area type'
}

spec['intersect_channel'] = {
    'num_classes'   : 2,
    'needs_binning' : False,
    'csv_index'     : 'Intersection channelisation'
}

spec['curvature'] = {
    'num_classes'   : 4,
    'needs_binning' : False,
    'csv_index'     : 'Curvature'
}

spec['upgrade_cost'] = {
    'num_classes'   : 3,
    'needs_binning' : False,
    'csv_index'     : 'Upgrade cost'
}

spec['drive_side_land_use'] = {
    'num_classes'   : 7,
    'needs_binning' : False,
    'csv_index'     : 'Land use - driver-side'
}

spec['passen_side_land_use'] = {
    'num_classes'   : 7,
    'needs_binning' : False,
    'csv_index'     : 'Land use - passenger-side'
}

spec['median_type'] = {
    'num_classes'   : 15,
    'needs_binning' : False,
    'csv_index'     : 'Median type'
}

spec['roadside_driver_side_distance'] = {
    'num_classes'   : 4,
    'needs_binning' : False,
    'csv_index'     : 'Roadside severity - driver-side distance'
}

spec['roadside_driver_side_object'] = {
    'num_classes'   : 17,
    'needs_binning' : False,
    'csv_index'     : 'Roadside severity - driver-side object'
}

spec['roadside_passenger_side_distance'] = {
    'num_classes'   : 4,
    'needs_binning' : False,
    'csv_index'     : 'Roadside severity - passenger-side distance'
}

spec['roadside_passenger_side_object'] = {
    'num_classes'   : 17,
    'needs_binning' : False,
    'csv_index'     : 'Roadside severity - passenger-side object'
}

spec['driver_side_paved_shoulder'] = {
    'num_classes'   : 4,
    'needs_binning' : False,
    'csv_index'     : 'Paved shoulder - driver-side'
}

spec['passenger_side_paved_shoulder'] = {
    'num_classes'   : 4,
    'needs_binning' : False,
    'csv_index'     : 'Paved shoulder - passenger-side'
}

spec['intersection_road_volume'] = {
    'num_classes'   : 7,
    'needs_binning' : False,
    'csv_index'     : 'Intersecting road volume'
}

spec['intersection_quality'] = {
    'num_classes'   : 3,
    'needs_binning' : False,
    'csv_index'     : 'Intersection quality'
}

spec['num_lanes'] = {
    'num_classes'   : 6,
    'needs_binning' : False,
    'csv_index'     : 'Number of lanes'
}

spec['lane_width'] = {
    'num_classes'   : 3,
    'needs_binning' : False,
    'csv_index'     : 'Lane width'
}

spec['curve_quality'] = {
    'num_classes'   : 3,
    'needs_binning' : False,
    'csv_index'     : 'Quality of curve'
}

spec['road_condition'] = {
    'num_classes'   : 3,
    'needs_binning' : False,
    'csv_index'     : 'Road condition'
}

spec['vehicle_parking'] = {
    'num_classes'   : 3,
    'needs_binning' : False,
    'csv_index'     : 'Vehicle parking'
}

spec['passenger_side_sidewalk'] = {
    'num_classes'   : 7,
    'needs_binning' : False,
    'csv_index'     : 'Sidewalk - passenger-side'
}

spec['drivers_side_sidewalk'] = {
    'num_classes'   : 7,
    'needs_binning' : False,
    'csv_index'     : 'Sidewalk - driver-side'
}

spec['bicycle_facilities'] = {
    'num_classes'   : 7,
    'needs_binning' : False,
    'csv_index'     : 'Facilities for bicycles'
}

def get_num_classes():
  num_classes = {}
  num_classes['sr'] = 5

  for key in spec.keys():
    num_classes[key] = spec[key]['num_classes']

  return num_classes
