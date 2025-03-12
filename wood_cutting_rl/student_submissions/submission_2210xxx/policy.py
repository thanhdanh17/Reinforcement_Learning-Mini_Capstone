import random
from abc import abstractmethod
import numpy as np

class Policy:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, observation, info):
        pass

    def _get_wood_size(self, wood):
        wood_w = np.sum(np.any(wood != -2, axis=1))
        wood_h = np.sum(np.any(wood != -2, axis=0))
        return wood_w, wood_h

    def _can_cut(self, wood, position, cut_size):
        cut_x, cut_y = position
        cut_w, cut_h = cut_size
        return np.all(wood[cut_x : cut_x + cut_w, cut_y : cut_y + cut_h] == -1)

class RandomCuttingPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_cuts = observation["cuts"]
        cut_size = [0, 0]
        wood_idx = -1
        cut_x, cut_y = 0, 0

        for cut in list_cuts:
            if cut["quantity"] > 0:
                cut_size = cut["size"]
                cut_x, cut_y = None, None

                for _ in range(100):
                    wood_idx = random.randint(0, len(observation["woods"]) - 1)
                    wood = observation["woods"][wood_idx]
                    wood_w, wood_h = self._get_wood_size(wood)
                    cut_w, cut_h = cut_size

                    if wood_w >= cut_w and wood_h >= cut_h:
                        cut_x = random.randint(0, wood_w - cut_w)
                        cut_y = random.randint(0, wood_h - cut_h)
                        if self._can_cut(wood, (cut_x, cut_y), cut_size):
                            break
                    
                if cut_x is not None and cut_y is not None:
                    break

        return {"wood_idx": wood_idx, "size": cut_size, "position": (cut_x, cut_y)}

class GreedyCuttingPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_cuts = observation["cuts"]
        cut_size = [0, 0]
        wood_idx = -1
        cut_x, cut_y = 0, 0

        for cut in list_cuts:
            if cut["quantity"] > 0:
                cut_size = cut["size"]
                for i, wood in enumerate(observation["woods"]):
                    wood_w, wood_h = self._get_wood_size(wood)
                    cut_w, cut_h = cut_size
                    
                    if wood_w >= cut_w and wood_h >= cut_h:
                        cut_x, cut_y = None, None
                        for x in range(wood_w - cut_w + 1):
                            for y in range(wood_h - cut_h + 1):
                                if self._can_cut(wood, (x, y), cut_size):
                                    cut_x, cut_y = x, y
                                    break
                            if cut_x is not None and cut_y is not None:
                                break
                        if cut_x is not None and cut_y is not None:
                            wood_idx = i
                            break
                
                if cut_x is not None and cut_y is not None:
                    break

        return {"wood_idx": wood_idx, "size": cut_size, "position": (cut_x, cut_y)}
