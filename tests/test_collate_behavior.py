# collate.py
import os, torch, random, numpy as np

class TileCollate:
    def __init__(self, base_seed: int, rank: int):
        self.base_seed = base_seed
        self.rank = rank
        self.epoch = 0
        self.g = torch.Generator()    # torch RNG owned by collate
        self._batch_ctr = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        # reseed collate's RNG stream deterministically (per epoch, per rank)
        self.g.manual_seed(self.base_seed + 10_000 * epoch + self.rank)
        self._batch_ctr = 0

    def __call__(self, batch):
        # advance the stream once per batch so batches differ deterministically
        bump = torch.randint(0, 2**31, (1,), generator=self.g).item()
        np.random.seed(bump % 2**32); random.seed(bump)

        # Example: use the RNG to shuffle/select tiles
        # (replace this with your real tiling logic)
        idx = torch.randperm(len(batch), generator=self.g).tolist()  # <-- uses self.g
        print(f"collate(pid={os.getpid()} epoch={self.epoch} ctr={self._batch_ctr}) idx={idx}")
        self._batch_ctr += 1
        return [batch[i] for i in idx]

from torch.utils.data import DataLoader, Dataset

class DS(Dataset):
    def __len__(self): return 8
    def __getitem__(self, i): return i

def main():
    seed, rank = 42, 0
    collate = TileCollate(seed, rank)
    loader = DataLoader(DS(), batch_size=4, num_workers=2,
                        persistent_workers=True, collate_fn=collate)

    for epoch in range(3):
        collate.set_epoch(epoch)              # <-- now this *changes* collate behavior
        for _ in loader:
            pass
        
if __name__ == "__main__":
    main()
