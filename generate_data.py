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
CENTER = [.5, .5]
BUTTON_SIZE = 0.1
END_STUDY_BUTTON_SIZE = .09375

ALLOW_TRANSLATION = False

# These are RGB values, they have a fixed luminance in HSL space.
COLORS = [
    [255, 0, 0],
    [255, 191, 0],
    [127, 255, 0],
    [0, 255, 255],
    [0, 63, 255],
    [127, 0, 255],
    [255, 0, 191]
]

ALL_OPTO_TYPES = ['E', 'Square']
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
    # print "init"
    # self.setup_images()
    self.setup_grid()
    self.setSize = set_size
    self.allowTranslation = False
    self.domains = DOMAINS
    self._gridStep = GRID['step']
    self._gridSize = GRID['size']

  def setup_images(self):
    self.images = {}
    h = BUTTON_SIZE * SCREEN_SIZE['height']
    w = BUTTON_SIZE * SCREEN_SIZE['width']

    self.images['greenImage'] = np.ones((h, w, 3))
    self.images['greenImage'][:, :, 0] = np.ones((h, w)) * 100
    self.images['greenImage'][:, :, 1] = np.ones((h, w)) *255
    self.images['greenImage'][:, :, 2] = np.ones((h, w)) *100

    self.images['redImage'] = np.zeros((h, w, 3))
    self.images['redImage'][:,:,0] = np.ones((h,w)) *255
    self.images['redImage'][:,:,1] = np.ones((h,w)) *100
    self.images['redImage'][:,:,2] = np.ones((h,w)) *100

    self.images['blackImage'] = np.zeros((h, w, 3))
    self.images['whiteImage'] = np.ones((h, w, 3)) * 255

    self.target_pixels = get_size_in_pixels(SCREEN_SIZE, TARGET_SIZE)
    self.target = np.ones((self.target_pixels['height'], self.target_pixels['width'], 3))

    # create self-paced endStudyPhase button image
    # borderSize = (h + w) / 8
    # TURQUOISE = [59, 165, 170]
    # images['turquoiseBox'] = np.zeros((h, w, 3))
    # borderFill(self.images.turquoiseBox, h, w, borderSize, TURQUOISE)

    # self.images.turquoiseImage = tensor.ByteTensor(h, w, 3)
    # for i in range(3):
    #   images.turquoiseImage:select(3, i):fill(TURQUOISE[i])

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
    # print location
    # print color
    # print self._array.shape
    for i in range(len(color)):
      self._array[int(location['bottom']):int(location['top']),
          int(location['left']):int(location['right']),i] = color[i]

  def _drawE(self, location, color, orientation):
    height = location['top'] - location['bottom']
    width = location['right'] - location['left']

    twentyPercentY = int(np.floor(.5 + 0.2 * height) + location['bottom'])
    fortyPercentY = int(np.floor(.5 + 0.4 * height) + location['bottom'])
    sixtyPercentY = int(np.floor(.5 + 0.6 * height) + location['bottom'])
    eightyPercentY = int(np.floor(.5 + 0.8 * height) + location['bottom'])

    twentyPercentX = int(np.floor(.5 + 0.2 * width) + location['left'])
    fortyPercentX = int(np.floor(.5 + 0.4 * width) + location['left'])
    sixtyPercentX = int(np.floor(.5 + 0.6 * width) + location['left'])
    eightyPercentX = int(np.floor(.5 + 0.8 * width) + location['left'])

    # fill with solid color
    self._drawSquare(location, color)

    # block out notches in the E by filling with background color
    if orientation == 'right':
      self._array[twentyPercentY: fortyPercentY,
          fortyPercentX: int(location['right']),: ] = BG_COLOR
      self._array[sixtyPercentY: eightyPercentY,
          fortyPercentX: int(location['right']),: ] = BG_COLOR
    elif orientation == 'left':
      self._array[twentyPercentY: fortyPercentY,
          int(location['left']):sixtyPercentX, : ] = BG_COLOR
      self._array[sixtyPercentY: eightyPercentY,
          int(location['left']):sixtyPercentX, : ] = BG_COLOR
    elif orientation == 'up':
      self._array[sixtyPercentY: int(location['top']),
          twentyPercentX:fortyPercentX,: ] = BG_COLOR
      self._array[sixtyPercentY: int(location['top']),
          sixtyPercentX: eightyPercentX,: ] = BG_COLOR
    elif orientation == 'down':
      self._array[int(location['bottom']):fortyPercentY,
          twentyPercentX:fortyPercentX,: ] = BG_COLOR
      self._array[int(location['bottom']):fortyPercentY,
          sixtyPercentX: eightyPercentX,: ] = BG_COLOR

  def drawObject(self, opt):
    if opt['optotype'] == 'Square':
      self._drawSquare(opt['location'], opt['color'])
    elif opt['optotype'] == 'E':
      self._drawE(opt['location'], opt['color'], opt['orientation'])


  def getStudyArrayData(self):
    self.domainType = random.choice(DOMAINS)
    domain = self.getDomain(self.domainType)

    studyArrayData = []

    # -- iterate over objects in the study array
    self._currentStudyLocationsSet = []
    for i in range(self.setSize):
      # -- generate random location, color, optotype, and orientation
      location = getRandomCoordinates(self._gridLimit, self._gridStep)
      color = random.choice(domain['colors'])
      index = domain['colors'].index(color)
      colorId = domain['colorIds'][index]
      optotype = random.choice(domain['optotypes'])
      orientation = random.choice(domain['orientations'])

      # -- make sure the random location was not already used
      while location in self._currentStudyLocationsSet:
        location = getRandomCoordinates(self._gridLimit, self._gridStep)
      self._currentStudyLocationsSet.append(location)

      studyArray = {
        'location' : location,
        'color' : color,
        'colorId' : colorId,
        'optotype' : optotype,
        'orientation' :orientation
      }
      studyArrayData.append(studyArray)

    return studyArrayData

  def getDomain(self, domainType):
    if domainType == 'E_ALL':
      return {
          'optotypes' : ['E'],
          'colors' : COLORS,
          'colorIds' : list(range(len(COLORS))),
          'orientations' : ORIENTATIONS
      }
    elif domainType == 'ALL':
       return {
          'optotypes' : ALL_OPTO_TYPES,
          'colors' : COLORS,
          'colorIds' : list(range(len(COLORS))),
          'orientations' : ORIENTATIONS
      }
    # elseif domainType == 'E_COLOR' then
    #   local fixedOrientation, _ = psychlab_helpers.randomFrom(ORIENTATIONS)
    #   return {
    #       optotypes = {'E'},
    #       colors = kwargs.colors,
    #       colorIds = psychlab_helpers.range(1, #kwargs.colors),
    #       orientations = {fixedOrientation}
    #   }
    # elseif domainType == 'E_ORIENTATION' then
    #   local fixedColor, fixedColorId = psychlab_helpers.randomFrom(kwargs.colors)
    #   return {
    #       optotypes = {'E'},
    #       colors = {fixedColor},
    #       colorIds = {fixedColorId},
    #       orientations = ORIENTATIONS
    #   }
    # elseif domainType == 'SQUARE_COLOR' then
    #   return {
    #       optotypes = {'Square'},
    #       colors = kwargs.colors,
    #       colorIds = psychlab_helpers.range(1, #kwargs.colors),
    #       orientations = ORIENTATIONS, -- orientation does nothing for a square
    #   }
    # elseif domainType == 'ALL' then
    #   return {
    #       optotypes = ALL_OPTO_TYPES,
    #       colors = kwargs.colors,
    #       colorIds = psychlab_helpers.range(1, #kwargs.colors),
    #       orientations = ORIENTATIONS
    #   }

  def getLegalTransforms(self, domainType, optotype):
    legalTransforms = []
    if domainType == 'E_ALL':
      legalTransforms = ['COLOR', 'ORIENTATION']
    # elseif domainType == 'E_COLOR' then
    #   legalTransforms = {'COLOR'}
    # elseif domainType == 'E_ORIENTATION' then
    #   legalTransforms = {'ORIENTATION'}
    # elseif domainType == 'SQUARE_COLOR' then
    #   legalTransforms = {'COLOR'}
    elif domainType == 'ALL':
      if optotype == 'E':
        legalTransforms = ['OPTOTYPE', 'COLOR', 'ORIENTATION']
      elif optotype == 'Square':
        legalTransforms = ['OPTOTYPE', 'COLOR']

    if self.allowTranslation:
      legalTransforms.append('TRANSLATION')

    return legalTransforms

  def getTestArray(self, studyArrayData):
    testArrayData = copy.deepcopy(studyArrayData)
    isNew = np.random.uniform(0, 1) > 0.5

    if not isNew:
      self.transform = 'NONE'
    else:
      # -- select an object to change
      changedObjectIndex = np.random.randint(1, len(testArrayData))
      #print changedObjectIndex
      # -- select a transformation to apply
      legalTransforms = self.getLegalTransforms(
          self.domainType,
          testArrayData[changedObjectIndex]['optotype']
      )
      self.transform = random.choice(
        legalTransforms)
      #print self.transform
      # -- apply the transformation
      if self.transform == 'COLOR':
        # -- make sure the random color is not the one that was already used
        newColor  = random.choice(COLORS)
        newColorId = COLORS.index(newColor)
        while newColor == studyArrayData[changedObjectIndex]['color']:
          newColor  = random.choice(COLORS)
          newColorId = COLORS.index(newColor)
        testArrayData[changedObjectIndex]['color'] = newColor
        testArrayData[changedObjectIndex]['colorId'] = newColorId
      elif self.transform == 'ORIENTATION':
        # -- make sure the random orientation is not the one that was already used
        newOrientation = random.choice(ORIENTATIONS)
        #print newOrientation
        #print "old "+studyArrayData[changedObjectIndex]['orientation']
        while newOrientation == studyArrayData[changedObjectIndex]['orientation']:
          newOrientation = random.choice(ORIENTATIONS) 
        #print newOrientation
        testArrayData[changedObjectIndex]['orientation'] = newOrientation

      elif self.transform == 'OPTOTYPE':
        if testArrayData[changedObjectIndex]['optotype'] == 'E':
          testArrayData[changedObjectIndex]['optotype'] = 'Square'
        elif testArrayData[changedObjectIndex]['optotype'] == 'Square':
          testArrayData[changedObjectIndex]['optotype'] = 'E'

      elif self.transform == 'TRANSLATION':
        # -- make sure the random location is not on top of another object
        newLocation = getRandomCoordinates(self._gridLimit,
          self._gridStep)
        while newLocation in self._currentStudyLocationsSet:
          newLocation = getRandomCoordinates(self._gridLimit, self._gridStep)

        testArrayData[changedObjectIndex]['location'] = newLocation
    return testArrayData, isNew

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
                      'optotype' : arrayData[i]['optotype'],
                      'orientation' : arrayData[i]['orientation']})

    return self._array


def zero_pad(array):
  new_array = np.zeros((84,84,3),dtype=np.int64)
  new_array[12:72, 12:72,:] = array
  return new_array


if __name__=='__main__':
  data_generator = DataGenerator(2)
  study_array = data_generator.getStudyArrayData()
  test_array, n = data_generator.getTestArray(study_array)
  data_size = 250000

  for i in range(data_size):
    study_array = data_generator.getStudyArrayData()
    test_array, n = data_generator.getTestArray(study_array)
    study_image = zero_pad(data_generator.renderArray(study_array))
    test_image = zero_pad(data_generator.renderArray(test_array))
    if n:
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave('/data/kvg245/1/'+"%07d" % i+'a.png',study_image)
        imsave('/data/kvg245/1/'+"%07d" % i+'b.png',test_image)
    else:
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave('/data/kvg245/0/'+"%07d" % i+'a.png',study_image)
        imsave('/data/kvg245/0/'+"%07d" % i+'b.png',test_image)
    if i%1000==0:
      print i
    
