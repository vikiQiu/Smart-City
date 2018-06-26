#!/usr/bin/env python  
from selenium import webdriver    
import time    
from selenium.webdriver.chrome.options import Options
import os
import re
import numpy as np

times = 1
# , "file:///Users/Viki/Documents/yhliu/Smart%20City/web/user_trail_tset.html"
urls = ["file:///Users/Viki/Documents/yhliu/Smart%20City/res/MapMatching_test/pre_cell/pre_trail.html"]


def get_pic(dir_name='pre_cell'):
    my_path = '../../res/MapMatching_test/' + dir_name
    js_file = my_path+'/' + dir_name.split('_')[0] + '_trail.html'
    files = [file for file in os.listdir(my_path) if '.js' in file]
    file_ind = [u'向' in file for file in files]
    for file in np.array(files)[file_ind]:
        trail = re.search(r'(线路.+?)_', file).group(1)
        if trail != '线路8':
            continue
        n = re.search(r'(.向.*?)_', file).group(1)
        pic_path = '../../pic/MapMatching_test'
        # pic_names = ['%s/%s/%s/%s.png' % (pic_path, trail, n, re.search(r'(.+?).js', file).group(1))]
        pic_names = []
        file_name = re.search(r'(.+?).js', file).group(1)
        print(file_name)
        pic_names.append('%s/%s/%s.png' % (pic_path, dir_name, re.search(r'(.+?).js', file).group(1)))
        pattern = re.compile('(wuhu(.+?).js)')
        content = re.sub(pattern, file, open(js_file).read())
        open(js_file, 'w').write(content)
        run_with_Chrome(urls[0], pic_names)


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def check_files_dir(files):
    for file in files:
        check_dir(os.path.dirname(file))


def run_with_PhantomJS(url):    
    common_step(webdriver.PhantomJS(executable_path=r'/Users/Viki/Downloads/softwares/phantomjs-2.1.1-macosx/bin/phantomjs'), url)    
    

def run_with_Chrome(url, pic_names):
    # chrome_options = Options()
    # chrome_options.add_argument("ignore-certificate-errors")
    common_step(webdriver.Chrome(executable_path=r'/Users/Viki/Downloads/softwares/chromedriver'), url, pic_names)


def common_step(driver, url, pic_names):    
    check_files_dir(pic_names)
    driver.get(url)    
    driver.set_window_size(1200, 900)
    for pic_name in pic_names:
        driver.save_screenshot(pic_name)
    driver.quit() 


if __name__ == '__main__':
    for i in range(times):
        print('=============Times %s============' % i)
        get_pic('pre_cell')


