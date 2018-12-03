import random
import numpy as np
import copy
from skimage.io import imsave
import warnings

seed = 7
np.random.seed(4)
random.seed(4)
SCREEN_SIZE = {'width' : 4, 'height' : 4}
BG_COLOR = [255]

TARGET_SIZE = .75  # fraction of screen to fill

GRID = {'size' : 64, 'step' : 8}

FIXATION_SIZE = .1
FIXATION_COLOR = [255, 0, 0] # RGB

COLORS = [
        0,128
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


class DataGenerator:

  def __init__(self, set_size):
    print "init"
    # self.setup_images()
    self.setSize = set_size

  def getStudyArrayData(self):
    studyArrayData = []

    self._currentStudyLocationsSet = []
    for i in range(self.setSize):
      # -- generate random location, color, optotype, and orientation
      location_x = np.random.randint(SCREEN_SIZE['width'])
      location_y = np.random.randint(SCREEN_SIZE['width'])
      location = [location_x, location_y]
      color = random.choice(COLORS)
      optotype = 'Square'
      # -- make sure the random location was not already used
      while location in self._currentStudyLocationsSet:
        location = getRandomCoordinates(self._gridLimit, self._gridStep)
      self._currentStudyLocationsSet.append(location)

      studyArray = {
        'location' : [location_x,location_y],
        'color' : color,
        'optotype' : optotype,
      }
      studyArrayData.append(studyArray)

    return studyArrayData

  def getTestArray(self, studyArrayData):
    testArrayData = copy.deepcopy(studyArrayData)
    isNew = np.random.uniform(0, 1) > 0.5

    if isNew:
      change_index = np.random.randint(self.setSize)
      new_color = random.choice(COLORS)
      while new_color == testArrayData[change_index]['color']:
        new_color = random.choice(COLORS)
      testArrayData[change_index]['color'] = new_color
    return testArrayData, isNew

  def renderArray(self, arrayData):
    self._array = np.zeros((int(SCREEN_SIZE['height']),
                             int(SCREEN_SIZE['width'])
                                      ), dtype=np.int64)
    self._array[:,:] = BG_COLOR[0]

    # -- draw objects
    for i in range(len(arrayData)):
      self._array[arrayData[i]['location'][0],arrayData[i]['location'][1]] = arrayData[i]['color']
    return self._array

if __name__=='__main__':
  data_generator = DataGenerator(2)
  study_array = data_generator.getStudyArrayData()
  test_array, n = data_generator.getTestArray(study_array)
  data_size = 10

  for i in range(data_size):
    study_array = data_generator.getStudyArrayData()
    test_array, n = data_generator.getTestArray(study_array)
    study_image = data_generator.renderArray(study_array)
    test_image = data_generator.renderArray(test_array)
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
