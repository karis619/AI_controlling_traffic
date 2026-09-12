# make_pedestrian_avatars.py
import os
from pathlib import Path
from rembg import remove
from PIL import Image, ImageOps

# ------------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------------
INPUT_FOLDER   = "pedestrians"      # put your original photos here
OUTPUT_FOLDER  = "pedestrians_raw"          # will be created automatically
AVATAR_W, AVATAR_H = 64, 128            # final size (feel free to change)
PADDING_COLOR = (0, 0, 0, 0)            # transparent

# ------------------------------------------------------------------
def ensure_folder(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def make_avatar(src_path: Path, dst_path: Path):
    # 1. Load
    img = Image.open(src_path).convert("RGBA")

    # 2. Remove background
    img_no_bg = remove(img)               # returns RGBA with transparent bg

    # 3. Resize while keeping aspect ratio
    img_resized = ImageOps.fit(img_no_bg, (AVATAR_W, AVATAR_H), Image.LANCZOS)

    # 4. Pad to exact size (transparent)
    canvas = Image.new("RGBA", (AVATAR_W, AVATAR_H), PADDING_COLOR)
    canvas.paste(img_resized, (0, 0), img_resized)   # alpha channel is respected

    # 5. Save
    canvas.save(dst_path, "PNG")
    print(f"saved {dst_path.name}")

# ------------------------------------------------------------------
def main():
    ensure_folder(OUTPUT_FOLDER)
    src_dir = Path(INPUT_FOLDER)

    if not src_dir.is_dir():
        print(f"Error: Folder '{INPUT_FOLDER}' not found. Create it and add images.")
        return

    for file in src_dir.iterdir():
        if file.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            out_file = Path(OUTPUT_FOLDER) / f"avatar_{file.stem}.png"
            make_avatar(file, out_file)

    print("\nAll avatars ready! Drop the folder into your simulation.")

if __name__ == "__main__":
    main()