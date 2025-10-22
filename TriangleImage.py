import numpy as np

class Boid:
   
    def __init__(
        self,
        height: float,
        width: float,
        position: Point,
        heading: float,
        velocity: Vector = None,
    ):
        
        self.height = height
        self.width = width
        self.position = position
        self.heading = heading
        self.color = (255, 255, 255)  # White color for the Boid
        self.velocity = velocity if velocity is not None else Vector([0, 0])
        self.vertices = self._compute_vertices()


    def _compute_vertices(self) -> List[tuple]:
      
        return [
            (self.position + self.height * Point([np.cos(self.heading), np.sin(self.heading)])).as_tuple(),
            (self.position + self.width * Point([-np.sin(self.heading), np.cos(self.heading)])).as_tuple(),
            (self.position + self.width * Point([np.sin(self.heading), -np.cos(self.heading)])).as_tuple()
        ]