from ImageReward import ReFL

if __name__ == "__main__":
    args = ReFL.parse_args()
    trainer = ReFL.Trainer("CompVis/stable-diffusion-v1-4", "data/refl_data.json", args=args)
    trainer.train(args=args)