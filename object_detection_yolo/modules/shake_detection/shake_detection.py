import numpy as np

class ShakeDetection:
    last_frame = None
    last_frames_zero_percentage = []
    starter_threshold = 0.4

    def is_hard_move(self, new_frame) -> bool:

        if self.last_frame is None or new_frame is None:
            return False
        
        out = self.last_frame - new_frame
        out[-11:11] = 0  # remove some random noise
        zeros = out.size - np.count_nonzero(out)
        size = out.size
        percentage_of_zeros = zeros / size
        self.last_frames_zero_percentage.append(percentage_of_zeros)
        q1 = np.percentile(self.last_frames_zero_percentage, 25)
        if len(self.last_frames_zero_percentage) >= 100:
            self.starter_threshold = q1
        # divided by 4 for hard move
        if percentage_of_zeros < (self.starter_threshold / 4):
            self.last_frames_zero_percentage = []
            return True
        else:
            self.updateChanges()

        return False
    
    def updateChanges(self):
        length_f = len(self.last_frames_zero_percentage)
        last100 = length_f - 102
        del self.last_frames_zero_percentage[last100:]
            