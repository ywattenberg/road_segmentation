from random import uniform
import requests
from PIL import Image
from PIL import ImageOps
import io
import numpy as np
from skimage.color import rgb2gray
from skimage.io import imsave



los_angeles = [
    {'n': 34.269260, 'w': -118.604202, 's': 34.171040, 'e': -118.370722},
    {'n': 34.100406, 'w': -118.362530, 's': 33.797995, 'e': -117.863483},
    {'n': 33.714559, 'w': -118.033473, 's': 33.636157, 'e': -117.746060}
]

chicago = [
    {'n': 42.072123, 'w': -88.311501, 's': 41.643560, 'e': -87.682533}
]

houston = [
    {'n': 29.875249, 'w': -95.563377, 's': 29.610542, 'e': -95.189842}
]

phoenix = [
    {'n': 33.688554, 'w': -112.381892, 's': 33.392095, 'e': -111.887507}
]

philadelphia = [
    {'n': 40.052889, 'w': -75.233393, 's': 39.904511, 'e': -75.140009},
    {'n': 40.049736, 'w': -75.144129, 's': 40.026079, 'e': -75.027399}
]

san_francisco = [
    {'n': 37.801910, 'w': -122.506267, 's': 37.737590, 'e': -122.398120},
    {'n': 37.826862, 'w': -122.295123, 's': 37.800282, 'e': -122.255984}
]

boston = [
    {'n': 42.387338, 'w': -71.141267, 's': 42.283792, 'e': -71.046510}
]
cities_boxes = [los_angeles, chicago, houston, phoenix, philadelphia, san_francisco, boston]

from random import uniform
import requests
from PIL import Image
from PIL import ImageOps
import io
import numpy as np
from skimage.color import rgb2gray

def pick_random_image_from_city(x, y, zoom_=18, width_=600, height_=600):
    
    url = "https://maps.googleapis.com/maps/api/staticmap?"
    center = "center=" + str(x) + "," + str(y)
    zoom = "&zoom="+str(zoom_)
    size = "&size="+str(width_)+"x"+str(height_)
    sat_maptype = "&maptype=satellite"
    road_maptype = "&maptype=roadmap"
    no_banners = "&style=feature:all|element:labels|visibility:off"
    api_key = "&key=" + "AIzaSyClkpTDJuocNePzROobsV6cAP_6NfzdwaA"

    sat_url = url + center + zoom + size + sat_maptype + no_banners + api_key
    road_url = url + center + zoom + size + road_maptype + no_banners + "&style=feature:road|element:geometry|color:0x000000" + api_key
    print(sat_url)
    print(road_url)
    req = requests.get(sat_url)
    print(req.status_code)
    sat_tmp = Image.open(io.BytesIO(req.content))
    #sat_tmp = Image.open(io.BytesIO(requests.get(sat_url).content))
    road_tmp = Image.open(io.BytesIO(requests.get(road_url).content))
    sat_image = np.array(sat_tmp.convert('RGB'))
    roadmap = np.array(road_tmp.convert('RGB'))

    mask = np.floor(rgb2gray(np.floor(roadmap/255))).astype(np.float32)
    new_mask = np.floor(rgb2gray(np.floor(roadmap >= 254))).astype(np.float32)
    third_mask_a = (roadmap[:, :, 0] == 255) & (roadmap[:, :, 1] == 235) & (roadmap[:, :, 2] == 161)
    third_mask_b = (roadmap[:, :, 0] == 255) & (roadmap[:, :, 1] == 242) & (roadmap[:, :, 2] == 175)

    mask = np.invert(np.array(road_tmp))
    new_mask = (mask[:, :] == 255) 

    
    return sat_image, mask, new_mask, roadmap

def get_center(image):
    xx = int(image.shape[0]/2)
    yy = int(image.shape[1]/2)
    return image[xx-200:xx+200, yy-200:yy+200]


def save_image(image, mask, counter):
    img_path = "data/additional_data/images/{}.png".format(counter)
    mask_path = "data/additional_data/masks/{}.png".format(counter)
    imsave(img_path, image)
    imsave(mask_path, (mask*255).astype(np.uint8))
    print("saved image at path: {}".format(img_path))
    print("saved mask at path: {}".format(mask_path))

if __name__ == "__main__":
    for i in range(1):
        city_nr = np.random.randint(len(cities_boxes)) #pick a city
        index = np.random.randint(len(cities_boxes[city_nr]))
        box = cities_boxes[city_nr][index]

        rand_x = uniform(box['w'], box['e'])
        rand_y = uniform(box['n'], box['s'])

        image, mask, new_mask, roadmap = pick_random_image_from_city(rand_y, rand_x)
        image = get_center(image)
        new_mask = get_center(new_mask)
        save_image(image, new_mask, i)