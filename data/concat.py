import cv2
import os
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import re
import logging
from typing import List, Tuple, Optional

class GridGenerator:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        grid_rows: int = 4,
        grid_cols: int = 6,
        file_pattern: str = r'frame(\d+)\.png$'
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.frames_per_grid = grid_rows * grid_cols
        self.file_pattern = file_pattern
        
        self.setup_logging()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def get_frame_number(self, filename: str) -> int:
        match = re.search(self.file_pattern, filename)
        return int(match.group(1)) if match else -1

    def load_and_validate_image(self, image_path: str) -> Optional[np.ndarray]:
        try:
            img = cv2.imread(image_path)
            if img is None:
                self.logger.warning(f"Failed to load image: {image_path}")
                return None
            return img
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            return None

    def create_grid(
        self,
        image_files: List[str],
        start_idx: int,
        grid_size: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        if not image_files:
            return None

        first_image = self.load_and_validate_image(image_files[0])
        if first_image is None:
            return None

        h, w = first_image.shape[:2]
        grid = np.zeros((h * grid_size[0], w * grid_size[1], 3), dtype=np.uint8)

        for idx, img_path in enumerate(image_files[start_idx:start_idx + self.frames_per_grid]):
            row = idx // grid_size[1]
            col = idx % grid_size[1]
            
            img = self.load_and_validate_image(img_path)
            if img is None:
                continue
                
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            
            grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = img

        return grid

    def process_directory(self, input_dir: Path) -> List[str]:
        obj_id = input_dir.name
        image_files = sorted(
            [str(f) for f in input_dir.glob('*.png')],
            key=self.get_frame_number
        )

        if not image_files:
            self.logger.warning(f"No images found in {input_dir}")
            return []

        saved_files = []
        num_grids = len(image_files) // self.frames_per_grid

        for grid_idx in range(num_grids):
            start_idx = grid_idx * self.frames_per_grid
            filename = f"{obj_id}_grid_{grid_idx + 1}.jpg"
            
            grid = self.create_grid(
                image_files,
                start_idx,
                (self.grid_rows, self.grid_cols)
            )
            
            if grid is not None:
                output_path = self.output_dir / filename
                cv2.imwrite(str(output_path), grid)
                saved_files.append(filename)
                self.logger.info(f"Saved grid: {filename}")

        return saved_files

    def process_all_directories(self):
        subdirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        self.logger.info(f"Processing {len(subdirs)} directories...")
        
        all_saved_files = []
        for subdir in tqdm(subdirs, desc="Processing directories"):
            saved_files = self.process_directory(subdir)
            all_saved_files.extend(saved_files)

        return all_saved_files

def main():
    parser = argparse.ArgumentParser(description="Generate grid layouts from video frames")
    parser.add_argument("--input_dir", required=True, help="Input directory containing frame folders")
    parser.add_argument("--output_dir", required=True, help="Output directory for grid images")
    parser.add_argument("--grid_rows", type=int, default=4, help="Number of rows in grid")
    parser.add_argument("--grid_cols", type=int, default=6, help="Number of columns in grid")
    parser.add_argument("--file_pattern", default=r'frame(\d+)\.png$', help="Regex pattern for frame filenames")
    
    args = parser.parse_args()

    generator = GridGenerator(
        args.input_dir,
        args.output_dir,
        args.grid_rows,
        args.grid_cols,
        args.file_pattern
    )
    
    generator.process_all_directories()

if __name__ == "__main__":
    main()
