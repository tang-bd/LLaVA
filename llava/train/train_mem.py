import os

from llava.train.train import train

os.environ["WANDB_PROJECT"]="mllm-guidance"
if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
