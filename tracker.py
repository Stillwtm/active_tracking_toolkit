import abc

class TrackerBase(abc.ABC):

    def __init__(self):
        pass
    
    @abc.abstractmethod
    def init(self, img, bbox):
        """initilize a tracker, using first frame and bounding box of the target

        Args:
            img (numpy array): 3xHxW, in RGB
            bbox (numpy array): (4,), [tlx, tly, w, h]

        Returns:
            None
        """        
        raise NotImplementedError

    @abc.abstractmethod
    def track(self, img):
        """get observed image and return bounding box

        Args:
            img (numpy array): 3xHxW, in RGB 

        Returns:
            bbox (numpy array): (4,), [tlx, tly, w, h]
        """
        raise NotImplementedError
