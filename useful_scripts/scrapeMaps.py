"""
scrapeMaps.py
Retrieve map data automatically

note that the selenium webdriver for mozilla is required to be in the project folder.

Download a copy at https://github.com/mozilla/geckodriver/releases
"""

# Selenium is a web driver wrapper
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# to check for alerts:
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# some essential python libraries to make it chooch
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import random
import time


def draw_map():
    """
    returns world_map: a Basemap object that contains the entire world. This will tell us whether or not a random GPS coordinate is on land
    """
    bottomlat = -89.0
    toplat = 89.0
    bottomlong = -170.0
    toplong =  170.0
    gridsize = 0.1
    world_map = Basemap(projection="merc", resolution = 'c', area_thresh = 0.1,llcrnrlon=bottomlong, llcrnrlat=bottomlat, urcrnrlon=toplong, urcrnrlat=toplat)
    world_map.drawcoastlines(color='black')
    return world_map


def main():
    # instantiate a chrome options object so you can set the size and headless preference
    # chrome_options = Options()
    # chrome_options.add_argument("--headless")
    # chrome_options.add_argument("--window-size=1920x1080")

    # download the chrome driver from https://sites.google.com/a/chromium.org/chromedriver/downloads and put it in the
    # current directory
    # chrome_driver = os.getcwd() +"\\chromedriver.exe"


    # To prevent download dialog
    profile = webdriver.FirefoxProfile()
    profile.set_preference('browser.download.folderList', 2) # custom location
    profile.set_preference('browser.download.manager.showWhenStarting', False)
    profile.set_preference('browser.download.dir', '/tmp')
    profile.set_preference('browser.helperApps.neverAsk.saveToDisk', 'text/csv')
    profile.set_preference("browser.download.dir", "C:\\Users\\Bret Nestor\\Downloads\\");
    profile.set_preference("browser.download.useDownloadDir", True);
    profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "zip");



    driver=webdriver.Firefox(profile)
    # driver = webdriver.Chrome(chrome_options=chrome_options, executable_path=chrome_driver)

    # capture the screen
    # driver.get_screenshot_as_file("capture.png")

    # navigate to the page you want to acquire data from
    driver.get("http://terrain.party/")
    search_button = driver.find_element_by_css_selector("[title=Search]")
    save_button = driver.find_element_by_css_selector("[title=Export]")
    smaller_scale = driver.find_element_by_css_selector("[title=Contract]")

    # make the scale smaller untilit is 8km large
    while True:
        scale=driver.find_element_by_css_selector("[class=scale]").text.split("\n")[0]
        print(scale+" km")
        if int(float(scale))==8:
            break
        smaller_scale.click()

    # instantiate map object
    world_map=draw_map()
    print("map is drawn")

    # try to acquire 50000 images
    for i in range(50000):
        while True:
            # # generate a random point that is on land
            lon, lat = random.uniform(-179,179), random.uniform(-89, 89)
            xpt, ypt = world_map( lon, lat ) # convert to projection map

            # Check if that point is on the map
            if world_map.is_land(xpt,ypt):
                # if it is on the map, print the name and break
                name="map_lon{:4.2f}_lat{:4.2f}".format(lon, lat) # note that the precision can be changed here.
                print(name)
                print("\n")
                break

        try:
            search_button.click() # click the search button on terrain.party
            search_alert = driver.switch_to_alert()
            print(search_alert.text)
            # enter gps coordinate string of land coordinates
            search_alert.send_keys("{}, {}".format(lat, lon))
            # click ok
            search_alert.accept()
            time.sleep(random.uniform(1,3)) # data should be pulled slowly

            # check if it can find that location
            try:
                WebDriverWait(driver, 1).until(EC.alert_is_present(),
                                               'Timed out waiting for PA creation ' +
                                               'confirmation popup to appear.')
                alert = driver.switch_to.alert
                alert.accept()
                continue
            except TimeoutException:
                # if it doesn't find the location we can continue
                pass

            # save the data to a zip folder by clicking the save button
            save_button.click()
            save_alert = driver.switch_to_alert()

            # enter name to save to in the popup
            name="map_lon{:4.2f}_lat{:4.2f}".format(lat, lon) # the precision can be saved here
            # click ok
            save_alert.send_keys(name)
            save_alert.accept()
            time.sleep(random.uniform(1,7)) # again.... download time is the holdup in this script

            # check if there is a problem saving
            try:
                WebDriverWait(driver, 1).until(EC.alert_is_present(),
                                               'Timed out waiting for PA creation ' +
                                               'confirmation popup to appear.')
                alert = driver.switch_to.alert
                # print(alert.text)
                alert.accept()
                continue
            except TimeoutException:
                # print("no alert")
                pass
        except:
            input("did not work") # break the script and try again




if __name__=="__main__":
    main()
