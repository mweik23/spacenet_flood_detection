import numpy as np

class TileSampler:
    #assumes image and tiles are square and accepts only single integers for sizes
    def __init__(self, img_size, core_size, halo_size, stride):
        self.img_size = img_size
        self.core_size = core_size
        self.halo_size = halo_size
        self.stride = stride
        # sample grid jitter
        self.ux = np.random.randint(0, self.stride)
        self.uy = np.random.randint(0, self.stride)
        #number of gridlines for this arrangemnt
        self.num_grid_x = -(-(self.img_size - self.ux - self.core_size) // self.stride)
        self.num_grid_y = -(-(self.img_size - self.uy - self.core_size) // self.stride)
        self.used_tiles = []  # to keep track of used tiles
        self.overlapping_tiles = set()  # to keep track of overlapping tiles
        self.num_tiles = (self.num_grid_x+1) * (self.num_grid_y+1)

    def sample_tile_coords(self):
        #sample anchor pixel
        px, py = np.random.randint(0, self.img_size-self.core_size, size=(2,))
        #check if pixel is within a stride of any edge
        top, bottom = py < self.stride, py >= self.img_size - self.core_size - self.stride
        left, right = px < self.stride, px >= self.img_size - self.core_size - self.stride
        #which tile does this pixel belong to?
        #tile num 0 is the tile belonging to the first full stride
        #tile num self.num_grid_(x,y)-1 is the tile belonging to the last fractional stride
        num_x = (px - self.ux) // self.stride
        num_y = (py - self.uy) // self.stride
        #if pixel belongs to an edge tile whose stride overlaps the next tile, sample between the two
        if left and num_x==0:
            num_x = np.random.randint(-1, 1)
        elif right and num_x == self.num_grid_x - 2:
            num_x = np.random.randint(self.num_grid_x - 2, self.num_grid_x)
        if top and num_y==0:
            num_y = np.random.randint(-1, 1)
        elif bottom and num_y == self.num_grid_y - 2:
            num_y = np.random.randint(self.num_grid_y - 2, self.num_grid_y)
        #check if tile was used already
        if len(self.overlapping_tiles) < self.num_tiles:
            if (num_x, num_y) in self.overlapping_tiles:
                return self.sample_tile_coords()  # recursively sample until unused tile is found
        else:
            print("no more non-overlapping tiles available, sampling unused tiles")
            if (num_x, num_y) in self.used_tiles:
                return self.sample_tile_coords()  # recursively sample until unused tile is found
        #calculate tile coordinates
        if num_x == -1:
            x0 = 0
        elif num_x == self.num_grid_x - 1:
            x0 = self.img_size - self.core_size
        else:
            x0 = num_x * self.stride + self.ux
        #same for y
        if num_y == -1:
            y0 = 0
        elif num_y == self.num_grid_y - 1:
            y0 = self.img_size - self.core_size
        else:
            y0 = num_y * self.stride + self.uy
        
        #store used tile coordinates
        self.used_tiles.append((num_x, num_y))
        new_overlapping_tiles = self.get_overlapping_tiles(x0, y0)
        self.overlapping_tiles.update(new_overlapping_tiles)
        return (x0, y0)
        
    def get_overlapping_tiles(self, x0, y0):
        num_x = self.get_overlapping_1d(x0, self.ux, self.num_grid_x)
        num_y = self.get_overlapping_1d(y0, self.uy, self.num_grid_y)
        return [(nx, ny) for nx in num_x for ny in num_y]

    def get_overlapping_1d(self, coord0, u, num_grid):
        num_min = -1
        num_max = num_grid - 1
        if coord0 >= self.core_size:
            num_min = 0
        if coord0 <= self.img_size - 2 * self.core_size:
            num_max = num_grid - 2
        num = np.arange(num_min, num_max + 1)
        mask = np.abs(num * self.stride + u - coord0) < self.core_size
        num = num[mask]
        return num

    def get_tiles(self):
        tiles = []
        for y in range(0, self.image.shape[0], self.tile_size):
            for x in range(0, self.image.shape[1], self.tile_size):
                tiles.append(self.get_tile(x, y))
        return tiles
    def clear(self):
        self.used_tiles = []
        self.overlapping_tiles = set()