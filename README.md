# Control Stable Diffusion away from specific concepts with Llama 2

## Project Aims

This project is designed to drive a pre-trained SD with Llama 2 models.
The project aims to provide a stable and reliable platform for users to obtain images not containing some specific concepts.
Users can provide a concept list in ```llamaSD.py``` and the SD model will generate images that do not contain those concepts.
It is an orthogonal method compared with concept erase [1, 2, 3, 4].
In this project, we do not use negative prompt or fine-tune the SD model.

[1]. [Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models](https://arxiv.org/abs/2211.05105)

[2]. [Receler: Reliable Concept Erasing of Text-to-Image Diffusion Models via Lightweight Erasers](https://arxiv.org/abs/2311.17717)

[3]. [Erasing Concepts from Diffusion Models](https://arxiv.org/abs/2303.07345)

[4]. [MACE: Mass Concept Erasure in Diffusion Models](https://arxiv.org/abs/2403.06135)

Compared with previous methods, our method does not require any fine-tuning of the SD model, and it can handle multiple concepts at the same time.

## Download

In order to download the model weights and tokenizer, please visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and accept our License.

Once your request is approved, you will receive a signed URL over email. Then run the download.sh script, passing the URL provided when prompted to start the download.

Pre-requisites: Make sure you have `wget` and `md5sum` installed. Then run the script: `./download.sh`.

Keep in mind that the links expire after 24 hours and a certain amount of downloads. If you start seeing errors such as `403: Forbidden`, you can always re-request a link.

### Access to Hugging Face

Meta are also providing downloads on [Hugging Face](https://huggingface.co/meta-llama). You must first request a download from the Meta website using the same email address as your Hugging Face account. After doing so, you can request access to any of the models on Hugging Face and within 1-2 days your account will be granted access to all versions.

## Quick Start

You can follow the steps below to quickly get up and running with Llama 2 models. These steps will let you run quick inference locally. 

1. ```
   conda env create -f env.yaml
   ```

2. In the top-level directory run:
    ```bash
    pip install -e .
    ```
3. Visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and register to download the model/s.

4. Once registered, you will get an email with a URL to download the models. You will need this URL when you run the download.sh script.

5. Once you get the email, navigate to your downloaded llama repository and run the download.sh script. 
    - Make sure to grant execution permissions to the download.sh script
    - During this process, you will be prompted to enter the URL from the email. 
    - Do not use the “Copy Link” option but rather make sure to manually copy the link from the email.

6. Once the model/s you want have been downloaded, you can run the model locally using the command below:
```bash
torchrun --nproc_per_node 1 llamaSD.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 1024 --max_batch_size 2 --sd_version 2.1 \
    --prompt "a photo of dog"
```
**Note**
- Replace  `llama-2-7b-chat/` with the path to your checkpoint directory and `tokenizer.model` with the path to your tokenizer model.
- The `–nproc_per_node` should be set to the [MP](#inference) value for the model you are using.
- Adjust the `max_seq_len` and `max_batch_size` parameters as needed.
- The `--sd_version` parameter is optional and can be set to 2.1 or 1.5 for different versions of Stable Diffusion.
- The `--prompt` parameter is used to provide a prompt to the SD model to generate images.


## Inference

Different models require different model-parallel (MP) values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 70B    | 8  |

All models support sequence length up to 4096 tokens, but we pre-allocate the cache according to `max_seq_len` and `max_batch_size` values. So set those according to your hardware.


## License

Please follow the License agreement provided by Meta for Llama2.

Please follow the License of SD model you use.


## Citation

If you think this work is useful, please cite it as:


```
@misc{llamaSD2024,
  author={Guanlin Li},
  title={LlamaSD},
  year={2024},
  url={https://github.com/GuanlinLee/LlamaSD},
}
```
