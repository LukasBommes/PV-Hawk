import filecmp
import os.path


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