class TiledImageDataset(Dataset):
    """
    Expects a list of image filepaths. Yields random tiles of size tile_size.
    Optionally returns (image, label) if you have labels per image.
    """
    def __init__(self, files, tile_size=512, transforms=None, max_tiles_per_image=4, labels=None):
        self.files = files
        self.tile = tile_size
        self.tfms = transforms
        self.max_tiles = max_tiles_per_image
        self.labels = labels  # e.g., dict[path] = class_idx or anything you need

        # Preload sizes to avoid opening in __getitem__
        self.shapes = []
        for f in self.files:
            with Image.open(f) as im:
                self.shapes.append(im.size)  # (W, H)

        # Virtual length: each image contributes several tiles per epoch
        self._len = len(self.files) * self.max_tiles

    def __len__(self): return self._len

    def __getitem__(self, i):
        img_idx = i // self.max_tiles
        path = self.files[img_idx]
        W, H = self.shapes[img_idx]
        t = self.tile

        # Sample a valid top-left for tile (if smaller than tile, will pad later)
        x0 = 0 if W <= t else random.randint(0, W - t)
        y0 = 0 if H <= t else random.randint(0, H - t)

        with Image.open(path) as im:
            im = im.convert("RGB")
            # Crop or pad to tile
            if W >= t and H >= t:
                im = im.crop((x0, y0, x0 + t, y0 + t))
            else:
                # pad small side(s) with reflection to keep content stats similar
                canvas = Image.new("RGB", (t, t))
                canvas.paste(im, (0, 0))
                im = canvas

        if self.tfms:
            im = self.tfms(im)

        y = None if self.labels is None else self.labels[path]
        return (im, y) if y is not None else im

# usage
from glob import glob
train_files = sorted(glob("data/train/**/*.jpg", recursive=True))
train_tiles = TiledImageDataset(train_files, tile_size=640, transforms=train_tfms, max_tiles_per_image=4)
train_loader = DataLoader(train_tiles, batch_size=32, shuffle=True, num_workers=num_workers,
                          pin_memory=True, prefetch_factor=2, persistent_workers=True, drop_last=True)