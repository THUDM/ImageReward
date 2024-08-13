import os
import subprocess
import sys
from pathlib import Path

import gradio as gr
import modules.images as images
import modules.scripts as scripts
import torch
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from modules import sd_samplers, shared
from modules.processing import (
    Processed,
    StableDiffusionProcessing,
    create_infotext,
    process_images,
)
from modules.shared import cmd_opts, opts, state

try:
    import ImageReward as reward
except ModuleNotFoundError as error:
    print('"image-reward" package has not been properly installed. Installing...')
    if subprocess.check_call(["pip", "install", "image-reward"]) == 0:
        print(f'"image-reward" package is successfully installed!')
        import ImageReward as reward
    else:
        print('"image-reward" package installation failed!')
        print(
            "Please open an issue with full error message at https://github.com/THUDM/ImageReward/issues"
        )


def unload_image_reward_model():
    del shared.image_reward_model


class Script(scripts.Script):
    def title(self):
        return "ImageReward - generate human preference scores"

    def show(self, is_txt2img):
        return True

    def ui(self, is_txt2img):
        with gr.Blocks():
            with gr.Row():
                gr.Markdown(
                    value="**Tip**: It will take a little time to **load** the ImageReward model before **the first generation**."
                )
            with gr.Row():
                with gr.Column():
                    filter_out_low_scores = gr.Checkbox(
                        value=False, label="Filter out images with low scores"
                    )
                with gr.Column():
                    lower_score_limit = gr.Textbox(value=0, label="Lower score limit")
            with gr.Row():
                gr.Markdown(
                    value="ImageReward model takes about **1,600 MB** of memory."
                )
            with gr.Row():
                unload_button = gr.Button(value="Unload Model From Memory")
                unload_button.click(unload_image_reward_model)
        return [filter_out_low_scores, lower_score_limit]

    def run(self, p, filter_out_low_scores, lower_score_limit):
        try:
            shared.image_reward_model  # if loaded, do nothing
        except AttributeError:
            # load the model
            if sys.platform == "win32":
                download_root = HUGGINGFACE_HUB_CACHE
            else:
                download_root = None
            print(f"Loading ImageReward model from {download_root}...")
            shared.image_reward_model = reward.load(
                "ImageReward-v1.0", download_root=download_root
            )  # using shared to make the model object global among modules

        # preprocess parameters
        if lower_score_limit != "":
            lower_score_limit = float(lower_score_limit)

        # generate images
        proc = process_images(p)

        # score
        gens = proc.images
        with torch.no_grad():
            for img in gens:
                score = shared.image_reward_model.score(p.prompt, img)
                img.info["score"] = score
                msg = f"ImageReward Score: {score:.4f}"
                if img.info.get("parameters") is None:
                    img.info["parameters"] = msg
                else:
                    img.info["parameters"] += "\n " + msg
                if getattr(proc, 'info', None) is None:
                    setattr(proc, 'info', msg)
                else:
                    proc.info += "\n " + msg

        # filter out images with scores lower than the lower limit
        if filter_out_low_scores:
            imgs = list(filter(lambda x: x.info["score"] > lower_score_limit, gens))
        else:
            imgs = gens

        # append score to info
        infotexts = [img.info["parameters"] for img in imgs]

        # sort to score
        img_info_list = list(zip(imgs, infotexts))
        img_info_list.sort(key=lambda x: x[0].info["score"], reverse=True)
        imgs, infotexts = list(zip(*img_info_list))

        # return Processed object
        return Processed(
            p=p,
            images_list=imgs,
            info=proc.info,
            seed=proc.seed,
            infotexts=infotexts,
            index_of_first_image=proc.index_of_first_image,
        )
