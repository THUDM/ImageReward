import os

from ImageReward import ReFL_SDXL

if __name__ == "__main__":
    args = ReFL_SDXL.parse_args()
    trainer = ReFL_SDXL.Trainer("stabilityai/stable-diffusion-xl-base-1.0", "data/refl_data.json", args=args)
    trainer.train(args=args)