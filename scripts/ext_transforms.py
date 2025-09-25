import collections
import torchvision
import torch
import torchvision.transforms.functional as F
import random 
import numbers
import numpy as np
from PIL import Image


class ExtRandomHorizontalFlip(object):
    """
    Randomly flips an image and its corresponding label horizontally.
    
    Attributes:
        p (float): The probability of the flip being applied.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        """
        Applies the transform to an image and its label.

        Args:
            img (PIL Image): The input image.
            lbl (PIL Image): The corresponding label mask.

        Returns:
            A tuple of (PIL Image, PIL Image): The transformed image and label.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(lbl)
        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ExtCompose(object):
    """
    A sequence of transformations to be applied to an image and its label.
    Works like torchvision.transforms.Compose but for both image and label.

    Attributes:
        transforms (list): A list of transform objects to apply in order.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl):
        """
        Applies the sequence of transforms.

        Args:
            img (PIL Image): The input image.
            lbl (PIL Image): The corresponding label mask.

        Returns:
            A tuple of (PIL Image, PIL Image): The transformed image and label.
        """
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ExtCenterCrop(object):
    """
    Crops the center of an image and its label to a specified size.

    Attributes:
        size (tuple): The desired output size (height, width) for the crop.
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, lbl):
        """
        Applies the center crop.

        Args:
            img (PIL Image): The input image.
            lbl (PIL Image): The corresponding label mask.

        Returns:
            A tuple of (PIL Image, PIL Image): The cropped image and label.
        """
        return F.center_crop(img, self.size), F.center_crop(lbl, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class ExtRandomScale(object):
    """
    Resizes an image and label by a random scaling factor.

    The scaling factor is chosen uniformly from a given range.

    Attributes:
        scale_range (tuple): A tuple (min_scale, max_scale) for the random scaling.
        interpolation: The interpolation method to use for resizing the image.
    """
    def __init__(self, scale_range, interpolation=Image.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        """
        Applies the random scaling.

        Args:
            img (PIL Image): The input image.
            lbl (PIL Image): The corresponding label mask.

        Returns:
            A tuple of (PIL Image, PIL Image): The rescaled image and label.
        """
        assert img.size == lbl.size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        target_size = ( int(img.size[1]*scale), int(img.size[0]*scale) )
        return F.resize(img, target_size, self.interpolation), F.resize(lbl, target_size, Image.NEAREST)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class ExtScale(object):
    """
    Resizes an image and label by a fixed scaling factor.

    Attributes:
        scale (float): The scaling factor to apply.
        interpolation: The interpolation method for image resizing.
    """
    def __init__(self, scale, interpolation=Image.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        """
        Applies the fixed scaling.

        Args:
            img (PIL Image): The input image.
            lbl (PIL Image): The corresponding label mask.

        Returns:
            A tuple of (PIL Image, PIL Image): The rescaled image and label.
        """
        assert img.size == lbl.size
        target_size = ( int(img.size[1]*self.scale), int(img.size[0]*self.scale) )
        return F.resize(img, target_size, self.interpolation), F.resize(lbl, target_size, Image.NEAREST)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class ExtRandomRotation(object):
    """
    Rotates an image and its label by a random angle.

    Attributes:
        degrees (tuple): The range of angles to choose from for the rotation.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """
        Selects a random angle from the specified degree range.
        
        Args:
            degrees (tuple): A tuple (min_angle, max_angle).
        
        Returns:
            (float): A random angle.
        """
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img, lbl):
        """
        Applies the random rotation.

        Args:
            img (PIL Image): The input image.
            lbl (PIL Image): The corresponding label mask.

        Returns:
            A tuple of (PIL Image, PIL Image): The rotated image and label.
        """
        angle = self.get_params(self.degrees)
        return F.rotate(img, angle, self.resample, self.expand, self.center), F.rotate(lbl, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string

class ExtRandomVerticalFlip(object):
    """
    Randomly flips an image and its corresponding label vertically.

    Attributes:
        p (float): The probability of the flip being applied.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        """
        Applies the transform to an image and its label.

        Args:
            img (PIL Image): The input image.
            lbl (PIL Image): The corresponding label mask.

        Returns:
            A tuple of (PIL Image, PIL Image): The transformed image and label.
        """
        if random.random() < self.p:
            return F.vflip(img), F.vflip(lbl)
        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class ExtPad(object):
    """
    Pads an image and label so their dimensions are divisible by a number.
    This can be useful to ensure compatibility with network architectures.

    Attributes:
        diviser (int): The number that the dimensions should be divisible by.
    """
    def __init__(self, diviser=32):
        self.diviser = diviser
    
    def __call__(self, img, lbl):
        """
        Applies padding to the image and label.

        Args:
            img (PIL Image): The input image.
            lbl (PIL Image): The corresponding label mask.
        
        Returns:
            A tuple of (PIL Image, PIL Image): The padded image and label.
        """
        h, w = img.size
        ph = (h//32+1)*32 - h if h%32!=0 else 0
        pw = (w//32+1)*32 - w if w%32!=0 else 0
        im = F.pad(img, ( pw//2, pw-pw//2, ph//2, ph-ph//2) )
        lbl = F.pad(lbl, ( pw//2, pw-pw//2, ph//2, ph-ph//2))
        return im, lbl

class ExtToTensor(object):
    """
    Converts a PIL Image or numpy array to a PyTorch tensor.
    The image is normalized to [0, 1], but the label is not.
    """
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type

    def __call__(self, pic, lbl):
        """
        Applies the conversion.

        Args:
            pic (PIL Image or numpy.ndarray): The input image.
            lbl (PIL Image or numpy.ndarray): The corresponding label mask.

        Returns:
            A tuple of (Tensor, Tensor): The converted image and label tensors.
        """
        if self.normalize:
            return F.to_tensor(pic), torch.from_numpy( np.array( lbl, dtype=self.target_type) )
        else:
            return torch.from_numpy( np.array( pic, dtype=np.float32).transpose(2, 0, 1) ), torch.from_numpy( np.array( lbl, dtype=self.target_type) )

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ExtNormalize(object):
    """
    Normalizes an image tensor using a given mean and standard deviation.
    The label tensor is passed through unchanged.
    
    Attributes:
        mean (list): The mean for each channel.
        std (list): The standard deviation for each channel.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, lbl):
        """
        Applies the normalization to the image tensor.

        Args:
            tensor (Tensor): The image tensor to normalize.
            lbl (Tensor): The label tensor (unaffected).

        Returns:
            A tuple of (Tensor, Tensor): The normalized image and original label.
        """
        return F.normalize(tensor, self.mean, self.std), lbl

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ExtRandomCrop(object):
    """
    Crops an image and label at a random location.

    Attributes:
        size (tuple): The desired output size (height, width) for the crop.
        padding (int): Padding to add to the image before cropping.
        pad_if_needed (bool): If true, pads the image if it's smaller than the crop size.
    """
    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """
        Calculates random crop coordinates.

        Args:
            img (PIL Image): The image to be cropped.
            output_size (tuple): The desired crop size (height, width).

        Returns:
            A tuple of (int, int, int, int): The top-left coordinates and size (i, j, h, w).
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, lbl):
        """
        Applies the random crop.

        Args:
            img (PIL Image): The input image.
            lbl (PIL Image): The corresponding label mask.

        Returns:
            A tuple of (PIL Image, PIL Image): The cropped image and label.
        """
        assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s'%(img.size, lbl.size)
        if self.padding > 0:
            img = F.pad(img, self.padding)
            lbl = F.pad(lbl, self.padding)

        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[1] - lbl.size[0]) / 2))

        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[0] - lbl.size[1]) / 2))

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class ExtResize(object):
    """
    Resizes an image and its label to a given size.

    Attributes:
        size (int or tuple): The desired output size.
        interpolation: The interpolation method for image resizing.
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.abc.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        """
        Applies the resizing.

        Args:
            img (PIL Image): The input image.
            lbl (PIL Image): The corresponding label mask.

        Returns:
            A tuple of (PIL Image, PIL Image): The resized image and label.
        """
        return F.resize(img, self.size, self.interpolation), F.resize(lbl, self.size, Image.NEAREST)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str) 
    
class ExtColorJitter(object):
    """
    Randomly changes the brightness, contrast, and saturation of an image.
    The label is not affected.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                         clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """
        Gets a randomized list of transforms to apply.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img, lbl):
        """
        Applies the color jittering to the image.

        Args:
            img (PIL Image): The input image.
            lbl (PIL Image): The label (unaffected).

        Returns:
            A tuple of (PIL Image, PIL Image): The transformed image and original label.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img), lbl

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class Lambda(object):
    """
    A simple wrapper to apply a lambda function as a transform.
    """
    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose(object):
    """
    A sequence of transformations to be applied to a single image.
    Used internally by ExtColorJitter.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
