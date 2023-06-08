"""
@File        : test-benchmark.py
@Description : Test ImageReward as a metric for text-to-image generation models.
@CreatedDate : 2023-04-17
@UpdatedDate : 2023-05-23
@Author(s)   : Jiazheng Xu <xjz22@mails.tsinghua.edu.cn>
               Shawn Yuxuan Tong <tongyuxuan361@gmail.com>
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

import ImageReward as RM


def test_benchmark(args):
    if torch.cuda.is_available():
        device = torch.device(
            f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cuda"
        )
    else:
        device = torch.device("cpu")

    # load prompt samples
    prompt_with_id_list = []
    with open(args.prompts_path, "r") as f:
        prompt_with_id_list = json.load(f)
    num_prompts = len(prompt_with_id_list)

    benchmark_types = args.benchmark.split(",")
    benchmark_types = [x.strip() for x in benchmark_types]

    # resolve the generation model list to be evaluated
    model_names = []
    if args.model == "default":
        model_names = ["openjourney", "sd2-1", "dalle2", "sd1-4", "vd", "cogview2"]
    elif args.model == "all":
        model_names = list(os.listdir(args.img_dir))
    else:
        model_names = args.model.split(",")
        model_names = [x.strip() for x in model_names]

    benchmark_name2result = {}
    for benchmark_type in benchmark_types:
        # load the model for benchmark
        print(f"Loading {benchmark_type} model...")

        model = None
        if benchmark_type == "ImageReward-v1.0":
            model = RM.load(name=benchmark_type, device=device, download_root=rm_path)
        else:
            model = RM.load_score(
                name=benchmark_type, device=device, download_root=rm_path
            )

        print(f"{benchmark_type} benchmark begins!")

        bench_scores = []

        result_dir = os.path.join(args.result_dir, benchmark_type)
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir, exist_ok=True)

        # evaluate the model(s)
        with torch.no_grad():
            for model_name in model_names:
                result_sample = []
                mean_score_list = []

                for item in tqdm(prompt_with_id_list):
                    prompt_id = item["id"]
                    prompt = item["prompt"]

                    model_img_dirpath = os.path.join(args.img_dir, model_name)
                    num_model_imgs = len(os.listdir(model_img_dirpath))

                    image_paths = [
                        os.path.join(model_img_dirpath, f"{prompt_id}_{x}.png")
                        for x in range(num_model_imgs // num_prompts)
                    ]
                    if benchmark_type == "ImageReward-v1.0":
                        rewards = model.score(prompt, image_paths)
                    else:
                        _, rewards = model.inference_rank(prompt, image_paths)

                    mean_score = np.mean(rewards)
                    mean_score_list.append(mean_score)

                    result_item = {
                        "id": prompt_id,
                        "prompt": prompt,
                        "mean_score": mean_score,
                        "rewards": rewards,
                    }
                    result_sample.append(result_item)

                result_path = os.path.join(
                    result_dir, f"{benchmark_type}-benchmark-{model_name}-result.json"
                )
                with open(result_path, "w") as f:
                    json.dump(result_sample, f, indent=4, ensure_ascii=False)

                bench_score = np.mean(mean_score_list)
                bench_scores.append(bench_score)

        model2score = dict(zip(model_names, bench_scores))
        benchmark_results_path = os.path.join(
            args.result_dir, f"{benchmark_type}-benchmark-results.json"
        )
        with open(
            benchmark_results_path,
            "w",
        ) as f:
            json.dump(model2score, f, indent=4, ensure_ascii=False)
        print(
            f"{benchmark_type} benchmark finished with result dumped to {benchmark_results_path}!"
        )

        benchmark_name2result[benchmark_type] = model2score

    max_len_model_name = max([len(model_name) for model_name in model_names])
    num_digits = 4
    for benchmark_type, model2score in benchmark_name2result.items():
        print(f"{benchmark_type} Benchmark Results:")
        print(f"|{'Model':^{max_len_model_name+2}s}|{'Score':^{num_digits+5}}|")
        print(f"|{'-'*(max_len_model_name+1)}:|:{'-'*(num_digits+3)}:|")
        for model_name, bench_score in model2score.items():
            print(
                f"| {model_name:>{max_len_model_name}s} | {bench_score:^{num_digits+3}.{num_digits}f} |"
            )
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompts_path",
        default="benchmark/benchmark-prompts.json",
        type=str,
        help="Path to the prompts json list file, each item of which is a dict with keys `id` and `prompt`.",
    )
    parser.add_argument(
        "--result_dir",
        default="benchmark/results/",
        type=str,
        help="Path to the metric results directory",
    )
    parser.add_argument(
        "--img_dir",
        default="benchmark/generations",
        type=str,
        help="Path to the generated images directory. The sub-level directory name should be the name of the model and should correspond to the name specified in `model`.",
    )
    parser.add_argument(
        "--model",
        default="default",
        type=str,
        help="""default(["openjourney", "sd2-1", "dalle2", "sd1-4", "vd", "cogview2"]), all or any specified model names splitted with comma(,).""",
    )
    parser.add_argument(
        "--benchmark",
        default="ImageReward-v1.0, CLIP",
        type=str,
        help="ImageReward-v1.0, Aesthetic, BLIP or CLIP, splitted with comma(,) if there are multiple benchmarks.",
    )
    parser.add_argument(
        "--rm_path",
        default=None,
        type=str,
        help="Path to place downloaded reward model in.",
    )
    parser.add_argument(
        "--gpu_id",
        default=None,
        type=str,
        help="GPU ID(s) to use for CUDA.",
    )

    args = parser.parse_args()

    if args.rm_path is not None:
        rm_path = os.path.expanduser(args.rm_path)
        if not os.path.exists(rm_path):
            os.makedirs(rm_path)
    else:
        rm_path = None

    test_benchmark(args)
