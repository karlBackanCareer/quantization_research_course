import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_dataset(self):
        image_files = list(self.input_dir.glob('*.png')) + list(self.input_dir.glob('*.jpg'))
        if not image_files:
            raise Exception(f"No images found in {self.input_dir}")

        all_results = []
        # Finer scale gradations around critical points
        scale_factors = [0.85, 0.60, 0.45, 0.30]

        for img_path in tqdm(image_files):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            base_name = img_path.stem
            original_path = str(self.output_dir / f"{base_name}_original.jpg")
            cv2.imwrite(original_path, img)

            for scale in scale_factors:
                down_size = (int(w * scale), int(h * scale))
                # Use INTER_AREA for downscaling to minimize moiré patterns
                downscaled = cv2.resize(img, down_size, interpolation=cv2.INTER_AREA)
                # Use INTER_LANCZOS4 for upscaling to maintain quality
                upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LANCZOS4)

                processed_path = str(self.output_dir / f"{base_name}_processed_{scale}.jpg")
                cv2.imwrite(processed_path, upscaled)

                all_results.append({
                    "original_path": original_path,
                    "processed_path": processed_path,
                    "scale": scale,
                    "psnr": float(psnr(img, upscaled)),
                    "ssim": float(ssim(img, upscaled, channel_axis=2))
                })

        df = pd.DataFrame(all_results)
        df.to_csv(self.output_dir / 'metrics.csv', index=False)
        with open(self.output_dir / 'image_pairs.json', 'w') as f:
            json.dump(all_results, f)
        return df

    def plot_metrics(self, df):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        for orig_path in df['original_path'].unique():
            data = df[df['original_path'] == orig_path]
            plt.plot(data['scale'], data['psnr'], '-o', label=Path(orig_path).stem)
        plt.xlabel('Scale Factor')
        plt.ylabel('PSNR')
        plt.title('PSNR vs Scale Factor')
        plt.legend()

        plt.subplot(1, 2, 2)
        for orig_path in df['original_path'].unique():
            data = df[df['original_path'] == orig_path]
            plt.plot(data['scale'], data['ssim'], '-o', label=Path(orig_path).stem)
        plt.xlabel('Scale Factor')
        plt.ylabel('SSIM')
        plt.title('SSIM vs Scale Factor')
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_plot.png')
        plt.close()

def main():
    processor = ImageProcessor("set14", "public/processed_images")
    results_df = processor.process_dataset()
    processor.plot_metrics(results_df)

if __name__ == "__main__":
    main()