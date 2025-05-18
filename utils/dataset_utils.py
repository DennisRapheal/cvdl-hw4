import os, cv2, random, torch, re, glob
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset
from typing import List, Tuple

def _make_pairs(de_paths: List[str]) -> Tuple[List[str], List[str]]:
    cl_paths = []
    for p in de_paths:
        # 先取得檔名
        fname = os.path.basename(p)

        # 判斷是哪一類
        if 'rain-' in fname:
            cl_name = fname.replace('rain-', 'rain_clean-')
        elif 'snow-' in fname:
            cl_name = fname.replace('snow-', 'snow_clean-')
        else:
            cl_name = fname  # fallback

        # 最後 clean 圖路徑
        cl_path = p.replace('degraded', 'clean').replace(fname, cl_name)
        cl_paths.append(cl_path)

    return de_paths, cl_paths

class PromptDataset(Dataset):
    """只負責取資料，不切 train/val。"""
    def __init__(self,
                 de_paths: List[str],
                 patch_size: int = 128,
                 is_train: bool = True,
                 flip_prob: float = 0.5):
        self.de_paths, self.cl_paths = _make_pairs(de_paths)
        self.patch_size = patch_size
        self.is_train   = is_train
        self.flip_prob  = flip_prob if is_train else 0.0

    # ------- util -------
    def __len__(self): return len(self.de_paths)

    def _random_crop(self, img):
        h, w = img.shape[:2]; p = self.patch_size
        if h <= p or w <= p:
            return img
        y = random.randrange(0, h - p + 1)
        x = random.randrange(0, w - p + 1)
        return img[y:y+p, x:x+p]

    # ------- 讀取 -------
    def __getitem__(self, idx):
        de_path = self.de_paths[idx]
        cl_path = self.cl_paths[idx]

        # 根據檔名取得 prompt id
        if 'rain-' in os.path.basename(de_path):
            prompt_id = 0
        elif 'snow-' in os.path.basename(de_path):
            prompt_id = 1
        else:
            prompt_id = -1  # optional fallback, 可加 assert 檢查

        de = cv2.imread(de_path)[:, :, ::-1]  # BGR→RGB
        cl = cv2.imread(cl_path)[:, :, ::-1]

        if random.random() < self.flip_prob:
            de, cl = de[:, ::-1, :], cl[:, ::-1, :]

        if self.is_train and self.patch_size:
            de, cl = self._random_crop(de), self._random_crop(cl)

        de = to_tensor(de.astype('float32') / 255.)
        cl = to_tensor(cl.astype('float32') / 255.)
        name = os.path.basename(self.cl_paths[idx])
        # torch.tensor(prompt_id)
        return (name, idx), de, cl


def _numeric_sort(files):
    return sorted(files, key=lambda p: int(re.search(r'(\d+)', p).group(1)))

class PromptTestDataset(Dataset):
    """Test dataset：檔名即 idx；不裁剪、不讀 GT。"""
    EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')

    def __init__(self, opt):
        root = opt.test_dir
        if os.path.isdir(os.path.join(root, 'degraded')):
            root = os.path.join(root, 'degraded')

        self.paths = _numeric_sort(
            [f for f in os.listdir(root) if f.lower().endswith(self.EXT)]
        )
        self.root = root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        name = self.paths[i]                 # e.g. "0.png"
        img  = cv2.imread(os.path.join(self.root, name))[:, :, ::-1]
        tensor = to_tensor(img / 255.)
        return name, tensor
    
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./../hw4_realse_dataset/train',)
    parser.add_argument('--num_samples', type=int, default=5,)
    args = parser.parse_args()

    # 讀取 degraded 清單
    de_dir = os.path.join(args.train_dir, 'degraded')
    de_paths = _numeric_sort(
        [os.path.join(de_dir, f) for f in os.listdir(de_dir)
         if f.lower().endswith(PromptTestDataset.EXT)]
    )

    print(f"共讀入 {len(de_paths)} 張 degraded 圖片")

    # 初始化 Dataset
    dataset = PromptDataset(de_paths, patch_size=128, is_train=True)

    # 檢查前 num_samples 筆資料
    for i in range(min(args.num_samples, len(dataset))):
        try:
            (name, idx), de, cl = dataset[i]
            print(f"[{i}] {name} | degraded shape: {de.shape}, clean shape: {cl.shape}")
        except Exception as e:
            print(f"[ERROR @ idx={i}] {dataset.de_paths[i]}")
            print(e)
