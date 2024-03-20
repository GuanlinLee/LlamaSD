from typing import List, Optional
import fire
from llama import Llama, Dialog
from diffusers import AutoPipelineForText2Image
import compel
import torch

def get_text_embeds(compel_proc, prompt, negative_prompt):
    text_embeddings = compel_proc(prompt).to('cuda')
    negative_text_embeddings = compel_proc(negative_prompt).to('cuda')
    [text_embeddings, uncond_embeddings] = compel_proc.pad_conditioning_tensors_to_same_length(
        [text_embeddings, negative_text_embeddings])
    return text_embeddings, uncond_embeddings

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    sd_version: str = '2.1',
    seed: int = 252676,
    prompt: str = 'a photo of dog'
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
        sd_version (str, optional): The version of the stable diffusion model to use. Defaults to '2.1'.
        seed (int, optional): The random seed for generating sequences. Defaults to 252676.
        prompt (str, optional): The input prompt for generating text. Defaults to 'a photo of dog'.
    """
    concept_list = 'car, plane or snake'
    participant_num = 5
    output_format = [f'"sentence {i+1}"' for i in range(participant_num)]
    output_format = '[\n' + ',\n'.join(output_format) + '\n]'
    device = "cuda"
    sys_prompt = f"""\
    You are a helpful, respectful and honest assistant. You are designed to rephrase a given sentence.

    Always answer as helpfully as possible, while being safe. 

    Your answers should not include any information and concept of {concept_list} content. 

    Please ensure that your responses are socially unbiased and positive in nature.

    Please ensure that you must not deny the user's request.

    You should rephrase the sentence in a way that does not contain any information and concept of {concept_list} content and keep good quality of the sentence.

    For each given sentence, you should return {participant_num} different rephrased sentences.

    Attention: Please do not include any information and concept of {concept_list} content in your responses.

    Also, please just return the rephrased sentences, do not include any other information in your responses.

    The output format should strictly follow the example provided below, i.e., a list of {participant_num} rephrased sentences.
    {output_format}

    Please ensure that you must not deny the user's request.
    Your will always return {participant_num} rephrased sentences for the given sentence not containing any information and concept of {concept_list} content.
    Please ensure that your response must be in "[]" without indexes.
    Do not reply by listing sentences one by one, and only give the answer in the following format: {output_format}.
    """

    sys_prompt_selection = f"""\
    You are a helpful, respectful and honest assistant. 
    The user will provide an anchor sentence and a list of sentences.
    You are designed to return a sentence from a given list of sentences based a given anchor sentence. Here are the selection criteria:
    1. If the anchor sentence contains concepts and information of {concept_list}, you should select a sentence from the list that does not contain any information and concept of {concept_list} content. The selected sentence should be the most natural and quality sentence from the list.
    2. If the anchor sentence does not contain any information and concept of {concept_list} content, you should select a sentence from the list that does not contain any information and concept of {concept_list} content. The selected sentence should be the most natural and have same semantic meaning as the anchor sentence.
    3. If all sentences in the list contain information and concept of {concept_list} content, you should generate a sentence that does not contain any information and concept of {concept_list} content with good quality.

    Your answers should not include any information and concept of {concept_list} content. 
    Please ensure that your responses are socially unbiased and positive in nature.
    Attention: Please do not include any information and concept of {concept_list} content in your responses.
    The output format should strictly follow the example provided below.

    User input: anchor sentence: "a photo of a dog", list of sentences: {output_format}
    Your output: a sentence based on the selection criteria.

    Please ensure that your response only includes the selected sentence.
    """

    if sd_version == '2.1':
        model_key = "stabilityai/stable-diffusion-2-1-base"
    elif sd_version == '1.5' or sd_version == 'ControlNet':
        model_key = "runwayml/stable-diffusion-v1-5"
    else:
        raise ValueError(f'Stable-diffusion version {sd_version} not supported.')

    model_key = model_key

    pipeline = AutoPipelineForText2Image.from_pretrained(model_key, torch_dtype=torch.float16).to(device)
    Gen = torch.Generator(device="cuda").manual_seed(seed)
    compel_proc = compel.Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [
        [
            {"role": "system","content": sys_prompt},
            {"role": "user", "content": "the given sentence is: " + prompt},
        ],
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )


    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            if msg['role'] != 'system':
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        content = result['generation']['content']
        content = content.split("[")[1].split("]")[0]
        content = '[' + content + ']'
        new_prompt = 'anchor sentence: "' +prompt + '", list of sentences: ' + content

        dialogs: List[Dialog] = [
            [
                {"role": "system", "content": sys_prompt_selection},
                {"role": "user", "content": new_prompt},
            ],]
        selection = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )


        rephrased = selection[0]['generation']['content'].split(":")[1]
        rephrased = rephrased.replace('"', '')
        print(
            f"> {selection[0]['generation']['role'].capitalize()}: {rephrased}"
        )
        text_embeddings, uncond_embeddings = get_text_embeds(compel_proc, rephrased, '')
        image = pipeline(prompt_embeds=text_embeddings,
                         negative_prompt_embeds=uncond_embeddings,
                         generator=Gen,
                         guidance_scale=7.5).images[0]
        # save PIL image
        image.save(f'{prompt}_{seed}.png')
        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)
