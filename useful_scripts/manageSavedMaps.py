"""
manageSavedMaps.py

Unzips maps and moves them to the correct folder. It deletes all other downloaded map-like files to save space
"""

import zipfile
import os
import glob
import time


def retrieveFiles(path_to_zip_file, directory_to_extract_to):
    """
    extracts the zip file to a specified folder than removes the original zip file
    """
    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()

    # delete the zip ZipFile
    os.remove(path_to_zip_file)
    return

def main():
    monitor=os.path.abspath("C:\\Users\\*\\Downloads\\") # change this if you are on mac or linux

    extract_folder=os.path.abspath(os.path.join(os.getcwd(), "\\temp_scrapes\\"))
    images_folder=os.path.abspath(os.path.join(os.getcwd(), "\\saved_scrapes\\"))

    while True:
        time.sleep(10)
        zipped_files=glob.glob(os.path.join(monitor, "map_lon*.zip"))
        if zipped_files:
            for zipped_file in zipped_files:
                try:
                    retrieveFiles(zipped_file, extract_folder)
                except:
                    # if os.path.isfile(zipped_file):
                    #     os.remove(zipped_file)
                    continue

        png_files=glob.glob(os.path.join(extract_folder, "map_lon*Merged*.png"))
        if png_files:
            for png in png_files:
                os.rename(png, os.path.join(images_folder, os.path.split(png)[-1]))

        for the_file in os.listdir(extract_folder):
            file_path = os.path.join(extract_folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                pass

if __name__=="__main__":
    main()
