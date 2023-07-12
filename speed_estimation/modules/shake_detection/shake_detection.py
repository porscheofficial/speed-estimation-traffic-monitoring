import numpy as np


class ShakeDetection:
    def __init__(self, threshold: float = 0.4) -> None:
        self.last_frame = None
        self.last_frames_zero_percentage = list[float]
        self.threshold = threshold

    def is_hard_move(self, new_frame: np.ndarray) -> bool:
        """Detect if the perspective in the video changed.

        A change in perspective is determined by subtracting the previous frame from the current.
        If the change exceeds the threshold defined, a perspective change is detected.

        @param new_frame:
            The current frame the speed_estimation is looking at.

        @return:
            A bool is returned indicating if the camera perspective has changed.
        """

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
            self.threshold = q1

        # divided by 4 for hard move
        if percentage_of_zeros < (self.threshold / 4):
            self.last_frames_zero_percentage = []
            return True

        self.updateChanges()

        return False

    def updateChanges(self):
        length_f = len(self.last_frames_zero_percentage)
        last100 = length_f - 102
        del self.last_frames_zero_percentage[last100:]
