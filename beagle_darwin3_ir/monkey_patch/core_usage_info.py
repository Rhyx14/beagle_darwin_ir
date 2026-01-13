from typing import Tuple

class CoreUsageInfo:
    def __init__(self, coord: Tuple[int, int], dendrite_used: int, dendrite_size: int,
                 axon_used: int, axon_size: int):
        self.coord = coord
        self.dendrite_used = dendrite_used
        self.dendrite_size = dendrite_size
        self.axon_used = axon_used
        self.axon_size = axon_size

    def is_error(self) -> bool:
        if self.is_dendrite_overflow() or self.is_axon_overflow():
            return True
        return False
    
    def is_dendrite_overflow(self) -> bool:
        return self.dendrite_used > self.dendrite_size
    
    def is_axon_overflow(self) -> bool:
        return self.axon_used > self.axon_size
    
    def __str__(self) -> str:
        dendrite_usage_percent = self.dendrite_used / self.dendrite_size * 100
        axon_usage_percent = self.axon_used / self.axon_size * 100
        return (f"Core{self.coord}: "
                f"Dendrite(size/used)\t {self.dendrite_size}/{self.dendrite_used}\t{dendrite_usage_percent:.2f}%"
                f"\tAxon(size/used) \t{self.axon_size}/{self.axon_used}\t{axon_usage_percent:.2f}%")
    