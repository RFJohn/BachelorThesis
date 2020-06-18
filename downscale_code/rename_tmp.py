import os


def rename(folder, images):
    for i in range(16000, 18000):
        os.rename(f"../val-data/{folder}/{images}-{i:05d}.png", f"../val-data/{folder}/{images}-{i-16000:05d}.png")


if __name__ == "__main__":
    rename("control", "control")
    rename("half_shift", "half")
    rename("original", "original")
    rename("quarter_shift", "quarter")
    rename("random0_shift", "random0")
