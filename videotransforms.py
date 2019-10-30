import numpy as np
import numbers
import random
import cv2
import matplotlib.pyplot as plt


class RandomCrop(object):
  """Crop the given video sequences (t x h x w) at a random location.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """

  def __init__(self, size):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size

  @staticmethod
  def get_params(img, output_size):
    """Get parameters for ``crop`` for a random crop.
    Args:
        img (PIL Image): Image to be cropped.
        output_size (tuple): Expected output size of the crop.
    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    t, h, w, c = img.shape
    th, tw = output_size
    if w == tw and h == th:
      return 0, 0, h, w

    i = random.randint(0, h - th) if h != th else 0
    j = random.randint(0, w - tw) if w != tw else 0
    return i, j, th, tw

  def __call__(self, imgs):

    i, j, h, w = self.get_params(imgs, self.size)

    imgs = imgs[:, i:i + h, j:j + w, :]
    return imgs

  def __repr__(self):
    return self.__class__.__name__ + '(size={0})'.format(self.size)


class CenterCrop(object):
  """Crops the given seq Images at the center.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """

  def __init__(self, size):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size

  def __call__(self, imgs):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    t, h, w, c = imgs.shape
    th, tw = self.size
    i = int(np.round((h - th) / 2.))
    j = int(np.round((w - tw) / 2.))

    return imgs[:, i:i + th, j:j + tw, :]

  def __repr__(self):
    return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomHorizontalFlip(object):
  """Horizontally flip the given seq Images randomly with a given probability.
  Args:
      p (float): probability of the image being flipped. Default value is 0.5
  """
  def __init__(self, p=0.5):
    self.p = p

  def __call__(self, imgs):
    """
    Args:
        img (seq Images): seq Images to be flipped.
    Returns:
        seq Images: Randomly flipped seq images.
    """
    if random.random() < self.p:
      # t x h x w
      return np.flip(imgs, axis=2).copy()
    return imgs

  def __repr__(self):
    return self.__class__.__name__ + '(p={})'.format(self.p)


class ResizeShortSideAndCenterCrop(object):
  """Crops the given seq Images at the center.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """

  def __init__(self, scale_size, crop_size):
      self.scale_size = scale_size
      self.crop_size = crop_size

  @staticmethod
  def get_params(img_shape, output_size):
    """Get parameters for ``crop`` for a center crop.
    Args:
        img (PIL Image): Image to be cropped.
        output_size (tuple): Expected output size of the crop.
    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for center crop.
    """
    h, w, c = img_shape
    th, tw = output_size
    if w == tw and h == th:
      return 0, 0, h, w

    i = int(np.round((h - th) / 2.))
    j = int(np.round((w - th) / 2.))
    return i, j, th, tw

  def __call__(self, imgs):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    new_imgs = []
    min_h = np.inf
    min_w = np.inf
    for img in imgs:
        h, w, c = img.shape
        d = self.scale_size - min(w, h)
        sc = 1 + d / min(w, h)
        img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
        h, w, c = img.shape
        min_h = min(min_h, h)
        min_w = min(min_w, w)
        new_imgs.append(img)

    i, j, h, w = self.get_params((min_h, min_w, c), (self.crop_size, self.crop_size))
    for idx,img in enumerate(new_imgs):
        new_imgs[idx] = img[i:i + h, j:j + w, :]
    new_imgs = np.asarray(new_imgs, dtype=np.float32)

    #return torch.from_numpy(new_imgs.transpose([3, 0, 1, 2]))
    return new_imgs

  def __repr__(self):
    return self.__class__.__name__ + '(size={0})'.format(self.size)


class ResizeShortSideAndRandomCrop(object):
  """Crops the given seq Images at random position.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """

  def __init__(self, scale_size, crop_size):
      self.scale_size = scale_size
      self.crop_size = crop_size

  @staticmethod
  def get_params(img_shape, output_size):
    """Get parameters for ``crop`` for a random crop.
    Args:
        img (PIL Image): Image to be cropped.
        output_size (tuple): Expected output size of the crop.
    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    h, w, c = img_shape
    th, tw = output_size
    if w == tw and h == th:
      return 0, 0, h, w

    i = random.randint(0, h - th) if h != th else 0
    j = random.randint(0, w - tw) if w != tw else 0
    return i, j, th, tw

  def __call__(self, imgs):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    new_imgs = []
    min_h = np.inf
    min_w = np.inf
    for img in imgs:
        h, w, c = img.shape
        d = self.scale_size - min(w, h)
        sc = 1 + d / min(w, h)
        img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
        h, w, c = img.shape
        min_h = min(min_h, h)
        min_w = min(min_w, w)
        new_imgs.append(img)

    i, j, h, w = self.get_params((min_h, min_w, c), (self.crop_size, self.crop_size))
    for idx,img in enumerate(new_imgs):
        new_imgs[idx] = img[i:i + h, j:j + w, :]
    new_imgs = np.asarray(new_imgs, dtype=np.float32)

    #return torch.from_numpy(new_imgs.transpose([3, 0, 1, 2]))
    return new_imgs

  def __repr__(self):
    return self.__class__.__name__ + '(size={0})'.format(self.size)
