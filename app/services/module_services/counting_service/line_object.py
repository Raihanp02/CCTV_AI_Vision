from dataclasses import dataclass

@dataclass
class LineObject:
    """
    coordinate: [x1,y1,x2,y2]
    """
    coordinate_relative: list = None
    coordinate_absolute: list = None
    direction_left_to_right: bool = True

    def __post_init__(self):
        if (self.coordinate_relative is None) == (self.coordinate_absolute is None):
            raise ValueError("Exactly one coordinate type must be provided")

    @property
    def coordinate(self):
        return self.coordinate_relative or self.coordinate_absolute

    def to_absolute(self, w: int, h: int):
        if self.coordinate_absolute:
            return self.coordinate_absolute
        x1, y1, x2, y2 = self.coordinate_relative
        return [int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)]