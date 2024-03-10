import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
import tqdm
import copy


def _make_w_io_base(f, mode: str):
    """
    Definition:
        -A function to create a file object for writing. If the input is a string, it will create the file and return the file object.
    
    Args:
        -f: A string path to the location on disk.
        -mode: Mode for opening the file.
        
    Returns:
        -f: A file object for writing.
    """
    if not isinstance(f, io.IOBase):  
        # Get the directory name
        f_dirname = os.path.dirname(f) 
        if f_dirname != "":
            # Create the directory
            os.makedirs(f_dirname, exist_ok=True)
        # Open the file 
        f = open(f, mode=mode)
    # Return the file object
    return f


def _make_r_io_base(f, mode: str):
    """
    Definition:
        -A function to create a file object for reading. If the input is a string, it will open the file and return the file object.
    Args:
        -f: A string path to the location on disk.
        -mode: Mode for opening the file.
    Returns:
        -f: A file object for reading.
    """
    if not isinstance(f, io.IOBase):
        # Open the file in the specified mode (read mode)
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """
    Definition:
        - Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    # Create a file object for writing
    f = _make_w_io_base(f, mode)
    
    if isinstance(obj, (dict, list)):
        # If the object is a dictionary or list, write it to the file
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        # If the object is a string, write it to the file
        f.write(obj)
    else:
        # If the object is not a dictionary, list, or string, raise an error due to unexpected type
        raise ValueError(f"Unexpected type: {type(obj)}")
    # Close the file
    f.close()


def jload(f, mode="r"):
    """
    Definition:
        - Load a .json file into a dictionary.
    Args:
        f: A string path to the location on disk.
        mode: Mode for opening the file.
    Returns:
        jdict: A dictionary containing the contents of the .json file.
    """
    # Create a file object for reading
    f = _make_r_io_base(f, mode)
    # Load the .json file into a dictionary
    jdict = json.load(f)
    # Close the file
    f.close()
    # Return the dictionary
    return jdict


if __name__ == "__main__":
    """
    Test the jdump and jload functions wwith a mock example.
    """
    # Create a dictionary
    d = {"a": 1, "b": 2}
    # Write the dictionary to a .json file
    jdump(d, "test.json") 
    
    #print the json file
    with open("test.json", "r") as file:
        print(file.read()) #we expect to see {"a": 1, "b": 2}
    
    # Load the .json file into a dictionary
    d2 = jload("test.json")
    # Print the loaded dictionary: we expect it to be the same as the original dictionary: {"a": 1, "b": 2}
    print(d2)
    # Remove the .json file
    os.remove("test.json")
