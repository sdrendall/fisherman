import numpy
from fisherman import generators as gen
from itertools import cycle, izip
from skimage import feature, measure


def normalize_array(A):
    """
    Returns the given array as a float64 with values normalized from 0 to 1
    """
    B = A.astype(numpy.float64) - A.min()
    return B/B.max()


def normalize_by_channel(image):
    """
    Normalizes each channel of the given image independently

    The returned image is of type float64 with the values of each channel normalized from 0 to 1
    """
    if len(image.shape) < 3:
        return normalize_array(image)
    else:
        return map_to_axis(normalize_array, image, 2)


def median_normalize(A, dtype=numpy.float64):
    B = A.astype(dtype)
    return B/numpy.median(B)
    

def map_to_axis(func, A, axis):
    """
    Maps the given function to the given array along the specified axis
    """
    return numpy.concatenate(map(func, numpy.split(A, A.shape[axis], axis)), axis)


def mean_range_regularization(A, prior_mean=None, prior_range=None, axis=None):
    """
    Rescales the input array by subtracting the mean, and dividing by range/2

    i.e.
    (A - prior_mean)/(prior_range/2)
    """
    if prior_mean is None:
        prior_mean = A.mean()

    if prior_range is None:
        prior_range = A.max() - A.min()
    
    if axis is None:
        return (A - prior_mean)/(prior_range/2)
    else:
        return numpy.concatenate([
                math.mean_range_regularization(channel, prior_mean=prior_mean, prior_range=prior_range)

                for (channel, z, s) in izip(
                    numpy.split(A, A.shape[axis], axis),
                    cycle(gen.iter_wrap(zero)),
                    cycle(gen.iter_wrap(scale))
                )
            ],
            axis
        )


def get_corner_peaks_by_label(image, labels, **kwargs):
    """
    Wrapper for skimage.feature.corner_peaks that optimizes the labels feature

    indices feature is not supported
    """
    peaks = numpy.zeros_like(image, dtype=numpy.bool)
    for props in measure.regionprops(labels):
        chunk = crop_from_bounding_box(props.bbox, image)
        chunk_peaks = feature.corner_peaks(chunk, labels=props.image, **kwargs)
        modify_with_bounding_box(props.bbox, peaks, chunk_peaks)

    return peaks


def modify_with_bounding_box(bb, image, modify):
    """
    Sets the values within the specified bounding box in image to those in modify
    """
    image[bb[0]:bb[2], bb[1]:bb[3], ...] = modify[...]


def crop_from_bounding_box(bb, image):
    """
    Returns a slice of the given image defined by the bounding box (bb)
    """
    return image[bb[0]:bb[2], bb[1]:bb[3], ...]


def log_kernel(sigma=0.5, filter_size=5):
    ind_range = numpy.arange(-numpy.floor(filter_size/2.0), numpy.ceil(filter_size/2.0))
    X, Y = numpy.meshgrid(ind_range, ind_range)
    Rsq = (X**2 + Y**2).astype(numpy.float64)
    
    return (-1/(numpy.pi*sigma**4)) * (1 - Rsq/(2 * sigma**2)) * numpy.exp(-Rsq/(2 * sigma**2))

    
def mean(seq):
    """
    Computes the mean of a given sequence
    """
    return numpy.asarray(list(seq)).mean()


class IntSequenceSampler(object):
    
    def __init__(self, upper_bound, n_samples=None, resample_period=100):
        self.upper_bound = upper_bound
        self.resample_period = resample_period
        self.master_set = numpy.arange(upper_bound, dtype=numpy.int64)
        self.resample_count = 0

        if n_samples is None:
            self.n_samples = upper_bound
        else:
            self.n_samples = n_samples

        self.resample_sample_set()
        
    def get_samples(self):
        self.resample_count += 1
        if self.resample_count % self.resample_period == 0:
            self.resample_sample_set()
            return self.sample_set

        else:
            self.sample_set += numpy.random.randint(self.upper_bound)
            self.sample_set %= self.upper_bound
            return self.sample_set


    def resample_sample_set(self):
        self.sample_set = numpy.random.choice(
            self.master_set, 
            size=self.n_samples,
            replace=False
        )
