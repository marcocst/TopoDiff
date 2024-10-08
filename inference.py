import torch
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, LMSDiscreteScheduler
from my_model import unet_2d_condition
import json
from PIL import Image
from utils import compute_location_loss, Pharse2idx, draw_box, setup_logger
import hydra
import os
from tqdm import tqdm

def inference(device, unet, vae, tokenizer, text_encoder, prompt, locations, phrases, cfg, logger):


    logger.info("Inference")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Phrases: {phrases}")

    # Get Object Positions

    logger.info("Conver Phrases to Object Positions")
    object_positions = Pharse2idx(prompt, phrases)
    print(object_positions)

    # Encode Classifier Embeddings
    uncond_input = tokenizer(
        [""] * cfg.inference.batch_size, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    # Encode Prompt
    input_ids = tokenizer(
            [prompt] * cfg.inference.batch_size,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

    cond_embeddings = text_encoder(input_ids.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
    generator = torch.manual_seed(cfg.inference.rand_seed)  # Seed generator to create the inital latent noise

    latents = torch.randn(
        (cfg.inference.batch_size, 4, 64, 64),
        generator=generator,
    ).to(device)

    noise_scheduler = LMSDiscreteScheduler(beta_start=cfg.noise_schedule.beta_start, beta_end=cfg.noise_schedule.beta_end,
                                           beta_schedule=cfg.noise_schedule.beta_schedule, num_train_timesteps=cfg.noise_schedule.num_train_timesteps)

    noise_scheduler.set_timesteps(cfg.inference.timesteps)

    latents = latents * noise_scheduler.init_noise_sigma
    regression = torch.nn.Linear(1280, 2)
    regression = regression.cuda()
    regression.weight.data = regression.weight.data * 0
    regression.bias.data[:] = 0

    loss = torch.tensor(10000)

    for index, t in enumerate(tqdm(noise_scheduler.timesteps)):
        iteration = 0

        while loss.item() > cfg.inference.loss_threshold and iteration < cfg.inference.max_iter and index < cfg.inference.max_index_step:
            latents = latents.requires_grad_(True)
            latent_model_input = latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=cond_embeddings)

            # update latents with guidance
            loss = compute_location_loss(attn_map_integrated_mid, attn_map_integrated_up, locations=locations,
                                   object_positions=object_positions, regression=regression) * cfg.inference.loss_scale
            print(f"loss is : {loss.item()}")
            rw, rb, grad_cond = torch.autograd.grad(loss.requires_grad_(True), [regression.weight, regression.bias, latents], allow_unused=True)

            clr = noise_scheduler.sigmas[index] ** 2
            latents = latents - grad_cond * clr

            if rw is not None:
                rlr = 1e-4
                blr = 1e-4
                if hasattr(regression.weight, "prev_grad"):
                    mom = regression.weight.prev_grad * 0.9 + rw * 0.1
                    regression.weight.data = regression.weight.data - mom * rlr
                    
                    mom = regression.bias.prev_grad * 0.9 + rb * 0.1
                    regression.bias.data = regression.bias.data - mom * blr
                else:
                    regression.weight.data = regression.weight.data - rw * rlr
                    regression.bias.data = regression.bias.data - rb * blr

                regression.weight.prev_grad = rw
                regression.bias.prev_grad   = rb
                # regression.weight.data = regression.weight.data - rw * 0.001
                # regression.bias.data = regression.bias.data - rb * 0.001
            iteration += 1
            torch.cuda.empty_cache()

        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=text_embeddings)

            noise_pred = noise_pred.sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg.inference.classifier_free_guidance * (noise_pred_text - noise_pred_uncond)

            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            torch.cuda.empty_cache()

    with torch.no_grad():
        logger.info("Decode Image...")
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images


@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg):

    # build and load model
    with open(cfg.general.unet_config) as f:
        unet_config = json.load(f)
    unet = unet_2d_condition.UNet2DConditionModel(**unet_config).from_pretrained(cfg.general.model_path, subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.general.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.general.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.general.model_path, subfolder="vae")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet.to(device)
    text_encoder.to(device)
    vae.to(device)



    # ------------------ example input ------------------
    examples = {"prompt": "There is an apple , an orange , and a water cup on the table", #"A hello kitty toy is playing with a purple ball on the right.",
            "phrases": "apple|orange|cup",
            "locations": ["major", "left"],
            'save_path': cfg.general.save_path
            }

    # Prepare the save path
    if not os.path.exists(cfg.general.save_path):
        os.makedirs(cfg.general.save_path)
    logger = setup_logger(cfg.general.save_path, __name__)

    logger.info(cfg)
    # Save cfg
    logger.info("save config to {}".format(os.path.join(cfg.general.save_path, 'config.yaml')))
    OmegaConf.save(cfg, os.path.join(cfg.general.save_path, 'config.yaml'))

    # Inference
    pil_images = inference(device, unet, vae, tokenizer, text_encoder, examples['prompt'], examples['locations'], examples['phrases'], cfg, logger)

    # Save example images
    for index, pil_image in enumerate(pil_images):
        image_path = os.path.join(cfg.general.save_path, 'example_new{}.png'.format(index))
        logger.info('save example image to {}'.format(image_path))
        # draw_box(pil_image, examples['bboxes'], examples['phrases'], image_path)
        pil_image.save(image_path)

if __name__ == "__main__":
    main()
