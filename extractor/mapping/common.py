import os
from collections import defaultdict
import csv
import pickle
import json

from extractor.geotransforms import enu2geodetic
from extractor.common import get_immediate_subdirectories


def load_reconstructions(root_dir):
    reconstructions = []
    for cluster_dir in sorted(get_immediate_subdirectories(root_dir)):
        file = os.path.join(root_dir, cluster_dir, "reconstruction.json")
        
        if os.path.isfile(file):
            reconstructions.extend(json.load(open(file, "r")))
            
    assert len(reconstructions) > 0, "No reconstructions found. Run OpenSfM first."
    return reconstructions


def get_image_size(reconstructions):
    """Read image width and height from a list of reconstruction dicts."""
    camera = list(reconstructions[0]['cameras'].keys())[0]
    width = reconstructions[0]['cameras'][camera]['width']
    height = reconstructions[0]['cameras'][camera]['height']
    return width, height


def load_tracks(tracks_file):
    """Load Tracks CSV file."""
    tracks_per_frame = defaultdict(list)
    tracks_per_id = defaultdict(list)
    with open(tracks_file, newline='', encoding="utf-8-sig") as csvfile:  # specifying the encoding skips optional BOM
        # automatically infer CSV file format
        dialect = csv.Sniffer().sniff(csvfile.readline(), delimiters=",;")
        csvfile.seek(0)
        csvreader = csv.reader(csvfile, dialect)
        for row in csvreader:
            frame_name = row[0]
            mask_name = row[1]
            track_id = row[2]
            tracks_per_frame[frame_name].append((mask_name, track_id))
            tracks_per_id[track_id].append((frame_name, mask_name))
    return tracks_per_frame, tracks_per_id


def save_modules(module_corners, file):
    module_centers_ = {track_id: pts_3d[0, :].reshape(1, 3) for track_id, pts_3d in module_corners.items()}
    module_corners_ = {track_id: pts_3d[1:, :].reshape(-1, 3) for track_id, pts_3d in module_corners.items()}

    modules = {
        "corners": module_corners_,
        "centers": module_centers_
    }
    pickle.dump(modules, file)
    

def save_modules_geocoords(module_corners, gps_origin, file):
    """Save WGS-84 coordinates of PV modules in GeoJSON file."""
    module_centers_ = {track_id: pts_3d[0, :].reshape(1, 3) for track_id, pts_3d in module_corners.items()}
    module_corners_ = {track_id: pts_3d[1:, :].reshape(-1, 3) for track_id, pts_3d in module_corners.items()}    
    
    module_geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    for track_id in module_corners.keys():
        
        # module outlines as polygons
        polygons = []
        for corner in module_corners_[track_id][::-1]:  # polygon must be right-handed, i.e. corners are in bl, br, tr, tl order
            lat, lon, alt = enu2geodetic(*corner, *gps_origin)
            polygons.append([lon, lat, alt])
        # repeat first corner to form a closed polygon
        corner = module_corners_[track_id][-1]
        lat, lon, alt = enu2geodetic(*corner, *gps_origin)
        polygons.append([lon, lat, alt])
        
        module_geojson["features"].append(
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    polygons
                ]
            },
            "properties": {
                "track_id": track_id
            }
        })
        
        # module centers as points
        lat, lon, alt = enu2geodetic(*module_centers_[track_id].reshape(3,), *gps_origin)
        point = [lon, lat, alt]
        
        module_geojson["features"].append(
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": point
            },
            "properties": {
                "track_id": track_id
            }
        })
    json.dump(module_geojson, file)