import filecmp
import os.path
import json
import pickle
import yaml
import csv


temp_dir_prefix = "pv-drone-inspect-test-"


def dirs_equal(dir1, dir2, ignore_file_contents=False):
    """
    Compare two directories recursively.
    
    args:
        dir1 (`str`): First directory path
        dir2 (`str`): Second directory path
        ignore_file_contents (`bool`): If True file contents are not
            considered during comparison. Otherwise, file contents
            are considered.

    returns: 
        True if the directory trees are the same and there were no 
        errors while accessing the directories or files, False 
        otherwise.
    """
    dirs_cmp = filecmp.dircmp(dir1, dir2)
    if len(dirs_cmp.left_only) > 0:
        print("Files in {} only: {}".format(dir1, dirs_cmp.left_only))
        return False
    if len(dirs_cmp.right_only) > 0:
        print("Files in {} only: {}".format(dir2, dirs_cmp.right_only))
        return False
    if len(dirs_cmp.funny_files) > 0:
        print("Funny files in {} and {} only: {}".format(dir1, dir2, dirs_cmp.funny_files))
        return False
    if not ignore_file_contents:
        (_, mismatch, errors) =  filecmp.cmpfiles(
            dir1, dir2, dirs_cmp.common_files, shallow=False)
        if len(mismatch) > 0:
            print("Files in {} and {} that are not equal: {}".format(dir1, dir2, mismatch))
            return False
        if len(errors) > 0:
            print("Files in {} and {} that could not be compared: {}".format(dir1, dir2, errors))
            return False
    for common_dir in dirs_cmp.common_dirs:
        new_dir1 = os.path.join(dir1, common_dir)
        new_dir2 = os.path.join(dir2, common_dir)
        if not dirs_equal(new_dir1, new_dir2, ignore_file_contents):
            return False
    return True


def load_file(root, root_ground_truth, file_name):
    """Loads and returns file contents of a file named
    `file_name` from both a `root` directory and 
    `root_ground_truth` directory."""

    if os.path.splitext(file_name)[1] == ".pkl":
        with open(os.path.join(root, file_name), "rb") as file:
            content = pickle.load(file)
        with open(os.path.join(root_ground_truth, file_name), "rb") as file:
            content_ground_truth = pickle.load(file)

    elif os.path.splitext(file_name)[1] == ".json" or os.path.splitext(file_name)[1] == ".geojson":
        with open(os.path.join(root, file_name), "r") as file:
            content = json.load(file)
        with open(os.path.join(root_ground_truth, file_name), "r") as file:
            content_ground_truth = json.load(file)

    elif os.path.splitext(file_name)[1] == ".yml" or os.path.splitext(file_name)[1] == ".yaml":
        with open(os.path.join(root, file_name), "r") as file:
            content = yaml.safe_load(file)
        with open(os.path.join(root_ground_truth, file_name), "r") as file:
            content_ground_truth = yaml.safe_load(file) 

    elif os.path.splitext(file_name)[1] == ".csv":
        content = []
        with open(os.path.join(root, file_name), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                content.append(row)
        content_ground_truth = []
        with open(os.path.join(root_ground_truth, file_name), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                content_ground_truth.append(row)       

    else:
        raise ValueError("Unknown file format.")

    return content, content_ground_truth