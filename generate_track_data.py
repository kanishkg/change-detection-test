import random
import numpy as np
import copy
from skimage.io import imsave
import warnings

seed = 7
np.random.seed(4)
random.seed(4)
SCREEN_SIZE = {'width' : 80, 'height' : 80}
BG_COLOR = [255, 255, 255]

TARGET_SIZE = .75  # fraction of screen to fill

GRID = {'size' : 64, 'step' : 8}

FIXATION_SIZE = .1
FIXATION_COLOR = [255, 0, 0] # RGB

COLORS = [
    [255, 0, 0],
    [255, 191, 0],
    [127, 255, 0],
    [0, 255, 255],
    [0, 63, 255],
    [127, 0, 255],
    [255, 0, 191]
]

ALL_OPTO_TYPES = ['Square']
ORIENTATIONS = ['left', 'right', 'up', 'down']
DOMAINS = [
      'E_ALL',
#       'E_COLOR',
#       'E_ORIENTATION',
#       'SQUARE_COLOR',
      'ALL'
  ]


def get_array_median(array):
  if len(array) % 2 == 0:
    return array[np.ceil(len(array)/2).astype(np.int64)]
  else:
    return (array[np.floor(len(array)/2).astype(np.int64)]+array[np.ceil(len(array)/2).astype(np.int64)])/2

def get_size_in_pixels(screen_size, size):
  return {'width': np.floor(size*screen_size['width']),
      'height': np.floor(size*screen_size['height'])}

def getRandomCoordinates(limit, step):
  full_domain = np.arange(np.floor(step/2), limit, step).tolist()
  grid_center = get_array_median(full_domain)
  invalid_domain = [-2,-1,1,2]
  x = random.choice(full_domain)
  y = random.choice(full_domain)
  while (x in invalid_domain) and (y in invalid_domain):
    x = random.choice(full_domain)
    y = random.choice(full_domain)
  return x, y

class DataGenerator:

  def __init__(self, set_size):
    print "init"
    # self.setup_images()
    self.setup_grid()
    self.setSize = set_size
    self._gridStep = GRID['step']
    self._gridSize = GRID['size']


  def setup_grid(self):
    self._gridLimit = GRID['size']-GRID['step']
    self._gridStep = GRID['step']
    self.target_pixels_size = {
        'width' : TARGET_SIZE * SCREEN_SIZE['width'],
        'height' : TARGET_SIZE * SCREEN_SIZE['height']
      }
    self._hFactor = self.target_pixels_size['width'] / GRID['size']
    self._vFactor = self.target_pixels_size['height'] / GRID['size']

  def _drawSquare(self, location, color):
    for i in range(len(color)):
      self._array[int(location['bottom']):int(location['top']),
          int(location['left']):int(location['right']),i] = color[i]

  def drawObject(self, opt):
    self._drawSquare(opt['location'], opt['color'])


  def getStudyArrayData(self):
    studyArrayData = []

    self._currentStudyLocationsSet = []
    flag = 0
    for i in range(self.setSize):
      # -- generate random location, color, optotype, and orientation
      location = getRandomCoordinates(self._gridLimit, self._gridStep)
      if flag==0:
          color = [0,255,0]
          flag = 1
      else: 
          color = [0,0,0]

      optotype = 'Square'
      # -- make sure the random location was not already used
      while location in self._currentStudyLocationsSet:
        location = getRandomCoordinates(self._gridLimit, self._gridStep)
      self._currentStudyLocationsSet.append(location)

      studyArray = {
        'location' : location,
        'color' : color,
        'optotype' : optotype,
      }
      studyArrayData.append(studyArray)

    return studyArrayData

  def getTestArray(self, studyArrayData):
    testArrayData = copy.deepcopy(studyArrayData)
    isSame = np.random.uniform(0, 1) > 0.5

    if isSame:
      testArrayData[0]['color'] = [0,0,255]
    else:
      testArrayData[0]['color'] = [0,0,0]
      testArrayData[1]['color'] = [0,0,255]
    return testArrayData, isSame

  def renderArray(self, arrayData):
    self._array = np.zeros((int(self.target_pixels_size['height']),
                             int(self.target_pixels_size['width']),
                                      3), dtype=np.int64)
    self._array[:,:,:] = BG_COLOR[0]

    # -- draw objects
    for i in range(len(arrayData)):
      location = {
          'left' : arrayData[i]['location'][0] * self._hFactor,
          'right' : (arrayData[i]['location'][0] + GRID['step']) * self._hFactor,
          'bottom' : arrayData[i]['location'][1] * self._vFactor,
          'top' : (arrayData[i]['location'][1] + GRID['step']) *
            self._vFactor,
      }

      self.drawObject({'location' : location,
                      'color' : arrayData[i]['color'],
                      'optotype' : arrayData[i]['optotype']})

    return self._array

def zero_pad(array):
  new_array = np.zeros((84,84,3),dtype=np.int64)
  new_array[12:72, 12:72,:] = array
  return new_array



if __name__=='__main__':
  data_generator = DataGenerator(2)
  study_array = data_generator.getStudyArrayData()
  test_array, n = data_generator.getTestArray(study_array)
  data_size = 10

  for i in range(data_size):
    study_array = data_generator.getStudyArrayData()
    test_array, n = data_generator.getTestArray(study_array)
    study_image = zero_pad(data_generator.renderArray(study_array))
    test_image = zero_pad(data_generator.renderArray(test_array))
    if n:
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        imsave('./'+"%07d" % i+'1a.png',study_image)
        imsave('./'+"%07d" % i+'1b.png',test_image)
        # imsave('/data/kvg245/1/'+"%07d" % i+'a.png',study_image)
        # imsave('/data/kvg245/1/'+"%07d" % i+'b.png',test_image)
    else:
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave('./'+"%07d" % i+'0a.png',study_image)
        imsave('./'+"%07d" % i+'0b.png',test_image)
        # imsave('/data/kvg245/0/'+"%07d" % i+'a.png',study_image)
        # imsave('/data/kvg245/0/'+"%07d" % i+'b.png',test_image)
    if i%1000==0:
      print i
