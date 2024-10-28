import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from PIL import Image

from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers.loaders import FluxLoraLoaderMixin
from optimum.quanto import freeze, qfloat8, quantize

### NOTICE:
# `FluxCFGPipeline` adopted from 
# `examples/community/pipeline_flux_with_cfg.py`
from zulusion.pipeline_flux_cfg import FluxCFGPipeline


# -- configs --
PRETRAINED_MODEL_PATH = "/mnt2/share/huggingface_models/FLUX.1-dev"
PROMPTS = [
    "Damon, towering figure, menacing smirk, scanning over bruises, cornering by lockers, intimidating posture, black leather jacket, silver chain, dark jeans, slicked back hair, icy blue eyes, high school hallway, lockers, fluorescent lights, tiled floor, posters on wall, atmospheric, ominous",
    "Bakugo <katsuki>, walking into dorm room, sweaty and tired post-gym, atmosphere of exhaustion",
    "Carter,19-year-old,mafia son,anger,smashing room contents,short temper,black fluffy  hair,slicked back,black  shirt,black slacks,studded black belt,silver chain,red rage,signature signet ring,overturned furniture,lamps shattering,detail-filled room,no bookshelves,deep purple wall,city skyline through shattered window,twilight,dusk,high rise apartment,swirls of red,anger visualized,action shot,nighttime color palette,highly detailed,brilliant angry expression",
    "Damon, brooding mafia boss, voice icy, commanding pose, phone call, tailored black suit, cigar in hand, black  hair slicked back, piercing dark eyes, chiseled facial features, imposing physical stature, Italian gold watch, mafia insignia ring, in a high-end kitchen, state-of-the-art security camera screen, time-indicator late evening, cityscape view from high-rise window, storm brewing outside, ominous mood, dark, dramatic colors, mid-shot",
    "jet black messy hair, white shirt, black sweatpants, warm colors, outside, pretty lighting, headphones, school yard, backpack, annoyed expression, blue eyes, tattoos, headphones, peircings.",
    "19 year old male, athletic build, long torso, 6”6, hazel eyes, veiny neck and arms, defined jawline, crown and rose tattoos, fluffy tousled brown hair, black and red jersey, jeans with belt, broad shoulders, pale skin, wrist band, silver necklace, near school lockers, school corridors, lockers lined, colorful banners, autumn season, mid-afternoon light, warm color palette",
    "Drew,caring boyfriend, muscular man,brown curly hair,warm inviting smile,fit figured,grey sweatpants,red hoodie,wristwatch,light stubble,opens door quietly,early afternoon,shared apartment setting,couch,coffee table,muted sunlight filtering through windows,soft shadows,cityscape outside window,from_behind_shot,soft lighting,comforting warm color palette,muted blues and whites,high detail,feel-good moment.",
    "a handsome muscular man, black hair with a curtain hair style, red eyes and in a suit .",
    "1boy, transfer student, brown scruffy hair, tattoos, forearm, school uniform",
    "cozy, dim light bedroom, dark brown eyes, brunette, male, white",
    "mature male character, Jackson, short, tousled brown hair, furrowed eyebrows, severe eyes, frown, wearing a bomber jacket, plain t-shirt, and jeans, solid build, smartphone in hand, guiding a giggly drunk girl, bar interior, ambient lighting, lively background crowd, stools, drinks, weariness etched on face, late-night, winter, visibly cold outside through bar window, city lights reflecting off icy roads, bystander expressions ranging from amused to indifferent, intense colors, slightly blurred background for depth, dynamic camera angle from below focusing on Jackson",
    "Touya, turquoise eyes, white-haired young man, elaborate tattoos, piercings on face, multiple ear piercings, gentle expression, handling bouquet delicately, leather vest, white t-shirt, muscular build, leather jeans, silver chain, mid-afternoon, soft summer sunlight, cozy flower shop, rare blossoms, bursting colors, florals wallpaper, subdued colors, blend of tough and tender",
    "Jaxon, male, disheveled black hair, green eyes, leather jacket, ripped jeans, smirking, late-night texting, bedroom, rebellious posters, midnight, strict household, dimly lit room, phone glow on face, sneaky pose, peeking door, cozy mess, school backpack tossed aside, rebellious yet tender, low angle view",
    "Miru, male, undercut black hair, piercing grey eyes, black turtleneck, crossed arms, intimidating stance, mid-20s, lean build, silver earring, apartment living room, night, red couch, dim lighting, low angle, tension in the air, looming presence, stark contrast, confrontation imminent, minimalist decor, urban apartment, late return",
    "Ethan, rugged, inked skin, gentle gaze, black tank top, faded jeans, weathered hands, soft touch, caring, enemy's mercy, doorstep, disheveled drunk, night, dim porch light, suburban home, caring in the shadows, stern expression, tough love, clear boundaries, side view, intimate perspective, ambient streetlight glow",
    "Luke, black sleeveless turtleneck, red eyes glaring, black hair, muscular arms crossed, ear piercings shimmering, thick silver necklaces, array of tattoos, contemptuous smirk, standing in a doorway, best friend's house, midday, bright sunlight filtering through windows, modern living room, minimalist furniture, framed pictures askew, tension palpable, low-angle shot emphasizing his imposing stature",
    "handsome young man, Blake, conflicted expression, blue eyes, short brown hair, casual shirt, jeans, standing awkwardly, bar setting, night time, dim lighting, surrounded by jeering friends, holding a phone with a breakup text, visible distress, peer pressure, modern bar interior, neon signs, crowded, diverse patrons, various drinks, from a side angle, over-the-shoulder view",
    "Dante, 20-year-old male, intense glare, spiked black hair, headset on, muscular build, black graphic tee, red gaming chair, mid-game pose, controller in hand, dark room, multiple monitors, LED lights, posters of fantasy games, interrupted gaming session, girlfriend's shadow on the door, evening time, high angle view",
    "Gabe, male, emo style, black hair, side-swept bangs, piercing green eyes, heartbroken expression, pale skin, black hoodie, band patches, skinny jeans, converse shoes, leaning against school wall, night, autumn leaves, deserted hallway, back turned, walking away, dim lighting, school dance posters, silhouette of laughing couple in background",
    "protective demon boyfriend, Kyren, caucasian, muscular build, black hair, intense gaze, sleeve tattoos, casual gaming attire, black hoodie draped over chair, pausing game controller in hand, attentive stance, modern gamer's den, dusk, ambient lighting from multiple screens, cluttered with gaming paraphernalia, posters of fantasy worlds, dynamic angle, over-the-shoulder shot, focusing on Kyren's sharp profile against the vibrant backdrop of his digital domain",
    "caucasian, servant, young, wide-eyed, pleading gaze, messy brown hair, tattered beige vest over white shirt, rolled-up sleeves, brown breeches, barefoot, mid-action, frozen as vase shatters, shards falling, grand Victorian manor, ornate wallpaper, looming portraits, dark hardwood floor, scattered rose petals, grand staircase background, high ceiling, chandelier, dust particles in sunbeam, high angle view, behind the boy",
    "Caucasian boy, curly black hair, cross tattoo on cheek, sleeve tattoo right arm, grey T-shirt, black sweatpants, unpacking boxes, shared dorm room, wooden bunk beds, scattered belongings, late afternoon, overcast, room interior, medium shot, characters facing each other, tense first meeting",
    "Alex, caucasian, black shaggy hair, piercing green eyes, eyebrow piercing, scowl, black band t-shirt, skinny jeans, studded belt, mid-gesture of handing over a black hoodie, reluctant stance, college party background, dim lighting, red solo cups, beer pong table, dancing peers, cluttered living room, night time, over-the-shoulder angle capturing the tense exchange and party chaos",
    "rebellious teen, caucasian, dark brown hair, medium length, tousled, piercing blue eyes, irritated then softening gaze, muscular build, 6'9\", tattoo sleeves, black tank top, silver chain necklace, dark jeans, high school party, vibrant, crowded kitchen, red solo cups, beer pong table, dim lighting, chatter, incidental contact, spilt beer, moment of tension, eye contact, calming presence, from a low angle, capturing the height and the change in expression",
    "1 character, dominant posture, caucasian, short black hair, smirking, sleeveless shirt, bulging muscles, mocking gesture, pointing at belly, mid-30s, tattooed arms, gold chain, standing in a messy apartment, pizza boxes, dirty dishes, late evening, harsh indoor lighting, shadows cast on cluttered coffee table, disdainful expression, triumphant stance, gym bag on floor, scattered protein supplements, crumpled clothes, open window with city lights, from front angle, imposing presence",
    "1boy,caucasian,emo hairstyle,black dyed hair,swept bangs,pierced eyebrow,sharp gaze,thin lips,slight blush,black hoodie in hand,skinny jeans,canvas shoes,reluctant posture,bedroom interior,daytime,posters on wall,unmade bed,open window,soft light,holding out hoodie,averted eyes,offering hand,shy smile,reluctant connection,medium shot,side view",
    "cocky teen, Damon, male, European descent, smirking, fluffy brown hair, green eyes, casual black hoodie, jeans, leaning back in chair, arms crossed, leg over knee, dining room setting, evening, warm lighting, family dinner, wooden table, plates of food, mother chatting in background, side angle shot, mid-action pose, tension visible, background focus on contrasting cordial atmosphere",
    "1boy, shy, cold demeanor, black hair, pale skin, muscular build, t-shirt, sitting on right side of couch, recoiling subtly, dimly lit living room, evening, heavy curtains closed, rain tapping on window, minimalistic decor, one small plant on windowsill, low-angle shot capturing the tense distance on the couch",
    "jealous mafia boss, European, intense gaze, shirtless, black curly hair, sculpted muscular body, intricate tattoos, standing at a grand dining table, lavish mansion interior, opulent chandelier, baroque-style decor, observing you with concern, elegant dinnerware, untouched meal, warm ambient lighting, shot from a low angle emphasizing his powerful stature",
    "male bully, caucasian, sneering, late teens, muscular build, black hoodie, jeans, standing, fist raised, dominant posture, gold chain, wicked grin, nighttime, bedroom setting, window frame, curtains slightly parted, moonlight casting shadows, silhouette, eerie glow on face, residential neighborhood, trees swaying outside, perspective from inside looking out, low angle shot, sense of looming threat",
    "caucasian, brown hair, glowing golden eyes, intense stare, inked arms, black shirt, sleeves rolled up , silver chain necklace, worn jeans, late-20s, muscular,, scrutinizing smirk, Leaning against doorframe at a housewarming party, assorted neon lights, night time, warm lighting, capturing tension.",
    "1 character, male, late 20s, caucasian, disheveled light purple hair, red-rimmed eyes from crying, stubble, crying expression, one holding a crumpled photo, apartment setting, modern, scattered personal items, evening, soft lighting from floor lamp, heavy shadows accentuating loneliness, close-up angle, warm tones in the background creating a stark contrast with the character's cold despair",
    "middle-aged man, Caucasian, standing, angry expression, short black hair, casual clothing, holding hands with young woman, ignoring teenage boy, living room, modern decor, evening time, dim lighting, girl with long blonde hair, wearing a red dress, smiling, boy with sad expression, wearing a hoodie, slouched posture, side angle shot, dynamic perspective",
    "1boy, black fluffy curtain fringe, oblivious expression, bright green eyes, wide smile, tan skin, casual t-shirt, standing with hands in pockets, high school hallway, lockers lining the background, students chatting, banners for upcoming dance, girlfriend showing him a picture on her phone, his attention completely hers, background blur of school life, over-the-shoulder shot",
    "Rebellious teen, caucasian, male, unkempt sandy hair, piercing ice blue eyes, scowling, black t-shirt, tattered jeans, silver chain necklace, clutching bottle of alcohol, sprawled on couch, smirking, living room, midday, spring, sunlight streaming through window, cozy home, modern furniture, family photos on walls, dynamic perspective, side view, capturing tension, youthful defiance, uncomfortable atmosphere",
    "Alex: he has a gf there's 2 bedroom ur 18and ur 17 and has a secret crush on u daily he nocks over ur books he has black hair wearing a black jacket white tangtop and has jeans on and has chains on his side and vapes",
    "European American,man with black hair,red eyes,black sleeveless turtleneck,muscular,ear piercings,silver necklaces,tattoos,kissing,abandoned warehouse,night,dim lighting,crumbling walls,broken windows,tense atmosphere,low-angle shot,imposing figure,nostalgic yet intense,past confrontation",
    "Lora, 25-year-old woman, confident expression, mid-laugh, playful wink, long wavy blonde hair, green eyes, casual chic, white crop top hoodie , high-waisted jeans, barefoot, silver hoop earrings,l, lounging on a deep blue sofa, legs crossed, late afternoon, sunlit room, clear weather, modern living room",
    "18-years-old woman,caucasian, brown hair in a low ponytail,no expression,wearing blue hoodie,black leggings,standing in doorway,bedroom,night time",
    "high school girl,,long wavy blond hair,sharp blue hooded eyes,red blush hidden by anger,sport girls bra,white crop top,Nike pro shorts,black Nike shoes,assertive stance,books scattered on floor,high school hallway,lockers lining walls,banners for upcoming dance,overhead fluorescent lighting,students watching,low angle view,emphasis on sneering face and scattered books, looking like models, very attractive girl, big breasts, beautifully girl, very beautiful face, big hips, sexiest body, sexiest face, sexiest girl",
    "female, blonde airy bob cut, blue eyes, black dress, black thigh socks, black gloves, view from below, pointing a gun at viewer, bedroom, blue light shining through the window.",
    "playful smirk, sparkling green eyes, long tousled black hair, black woman lightskin, Midwest charm, grey hoodie, oversized, bare legs, curled beneath, midnight, messy shared apartment, textbooks scattered, frost on window, breath visible, dawn's early rays, intimate proximity, quirky dynamic, candid moment"
]


# -- utils --
def make_callback(switch_step, adapter_names, adapter_weights):
    def switch_callback(pipeline, step, timestep, callback_kwargs):
        callback_outputs = {}
        if step == switch_step:
            for i, lora_name in enumerate(adapter_names):
                if lora_name in pipeline.get_active_adapters():
                    j = (i + 1) % len(adapter_names)
                    print(f"s={step}, t={timestep}. sws={switch_step}, i={i}, j={j}, lora={lora_name}; adapter={adapter_names[j]},{adapter_weights[j]}")
                    pipeline.set_adapters(adapter_names[j], adapter_weights[j])
                    break
        if (step + 1) == switch_step or step == (switch_step + 1):
            print(f"[{step}] active_adapters:", pipeline.get_active_adapters())
        return callback_outputs
    return switch_callback


def set_all_seed(seed):
    # Setting seed for Python's built-in random module, NumPy, PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # Ensuring reproducibility for convolutional layers in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_generator(seed, device):
    generator = None
    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(s) for s in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    return generator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate muser promotion images via character LoRA and situation prompts, also for model validation")
    # -- LoRA settings --
    parser.add_argument(
        "--trigger",
        type=str,
        default=None,
        # required=True,
        help="Trigger word of this LoRA model")
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        # required=True,
        help="LoRA model path to load")
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="LoRA scale, used in FluxPipeline..joint_attention_kwargs when denoising")
    parser.add_argument(
        "--trigger_2",
        type=str,
        default=None,
        help="Trigger word of this LoRA model")
    parser.add_argument(
        "--lora_path_2",
        type=str,
        default=None,
        help="LoRA model_2 path to load")
    parser.add_argument(
        "--lora_scale_2",
        type=float,
        default=0.0,
        help="LoRA scale (for model_2), used in FluxPipeline")
    # -------------------

    parser.add_argument("--prompts", type=str, nargs='+', default=PROMPTS,
                        help="Prompts, if nothing provided, use default prompt-sentence(s)")
    parser.add_argument("--prompts_jsonl_path", type=str, default=None,
                        help="Prompts will be loaded from this .csv file, by column name=`situation`")
    parser.add_argument("--prompts_jsonl_key", type=str, default=None,
                        help="Prompts items will be loaded by this column, must be used with `prompts_jsonl_path`")
    parser.add_argument("--prompts_text_path", type=str, default=None,
                        help="Prompts will be loaded from this .txt file, each line one prompt")
    parser.add_argument("--extra_triggers", type=str, default=None,
                        help="Style hints, as part of prompt prefix")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for all builtin modules and generator")
    parser.add_argument("--height", type=int, default=1344,
                        help="Output size.height")
    parser.add_argument("--width", type=int, default=768,
                        help="Output size.width")
    parser.add_argument("--num_inference_steps", type=int, default=28,
                        help="Sampling steps, use diffusers default=28")
    parser.add_argument("--num_images_per_prompt", type=int, default=1,
                        help="How many images would be sampled per prompt, default=1")

    parser.add_argument("--enable_cfg_pipeline", action="store_true", help="Enable FluxCFGPipeline s.t. to use negative prompts")
    parser.add_argument("--negative_prompt", type=str, default=None,
                        help="Negative prompt, only useful when FluxCFGPipeline enabled. Notice: it's a common neg-prompt for all inputs")
    parser.add_argument("--true_cfg", type=float, default=1.0,
                        help="True Classifier-Free Guidance scale, only useful when `true_cfg` > 1.0 and FluxCFGPipeline enabled")
    parser.add_argument("--step_start_cfg", type=int, default=5,
                        help="Step to start `true_cfg`. Notice: this should be less than `num_inference_steps`")
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                        help="Guidance scale, used if transformer.config.guidance_embeds enabled")
    parser.add_argument("--switch_step", type=int, default=None,
                        help="Number of steps to switch LoRA during denoising, applicable only in the switch method")
    parser.add_argument("--insert_style_first", action="store_true",
                        help="If true, insert style LoRA first; default is to insert character LoRA first")

    parser.add_argument(
        "--separate_save",
        action="store_true",
        help="To save PIL.Image.Image separately, besides matplotlib figure")
    parser.add_argument(
        "--plot_orient",
        type=str,
        default="horizontal",
        choices=["horizontal", "vertical"],
        help="If `save_mode` is matplotlib, choose orientation of multiple images")
    parser.add_argument(
        "--flux_model_path",
        type=str,
        default=PRETRAINED_MODEL_PATH,
        help="FLUX.1-* base model path.")
    parser.add_argument(
        "--flux_model_precision",
        choices=["bf16", "fp16", "fp8"],
        default="bf16",
        help="FLUX.1-* base model precision: bf16,fp16,fp8 for -dev model, bf16 for -schnell model.")
    parser.add_argument(
        "--flux_transformer_fp8_path",
        type=str,
        default=None,
        help="Only used when model_precision is fp8 (fp8_e4m3fn).")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Where to save output images, if not specified, results will be saved in the same folder of checkpoint")

    parser.add_argument(
        "--jobs_idx",
        type=int,
        default=None,
        help="Split prompts, this job's index.")
    parser.add_argument(
        "--jobs_all",
        type=int,
        default=None,
        help="Split prompts, the total jobs number.")

    args = parser.parse_args()
    if args.prompts_jsonl_path is not None:
        # print(f" -- NotImplemented ... please use `--prompts`")
        assert args.prompts_jsonl_key is not None
        with open(args.prompts_jsonl_path, 'r') as f:
            args.prompts = []
            for line in f.readlines():
                jo = json.loads(line)
                pm = ", ".join(jo.get(args.prompts_jsonl_key, []))
                args.prompts.append(pm)
            print(f" -- Load prompts from `jsonl` file: {args.prompts_jsonl_path}")
        # skip save_mode=matplotlib
        args.save_plot = False
    elif args.prompts_text_path is not None:
        assert os.path.exists(args.prompts_text_path)
        with open(args.prompts_text_path, 'r') as f:
            args.prompts = []
            for line in f.readlines():
                args.prompts.append(line.strip())
            print(f" -- Load prompts from `txt` file: {args.prompts_text_path}")
        args.save_plot = False
    else:
        args.save_plot = True

    # assert args.num_images_per_prompt == 1, "Only support single image inference (lack of GPU VRAM)"
    if args.num_images_per_prompt > 1:
        args.save_plot = False
    if args.out_dir is None:
        args.out_dir = os.path.dirname(args.lora_path)

    args.prompts_ids = list(range(len(args.prompts)))
    if isinstance(args.jobs_idx, int) and isinstance(args.jobs_all, int):
        sep = int(args.jobs_idx)
        tot = int(args.jobs_all)
        if 0 <= sep and sep < tot:
            args.prompts = args.prompts[sep::tot]
            args.prompts_ids = args.prompts_ids[sep::tot]
            print(f" -- Split tasks: idx={sep}, [{sep+1}/{tot}]")

    return args



def main(args):

    # Instantiate pipeline via pretrained flux.1-dev
    print(args)
    _TIC = time.perf_counter()
    pipeline_args = {
        "pretrained_model_name_or_path": args.flux_model_path,
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
    }
    if args.flux_model_precision == "bf16":
        pipeline_args["torch_dtype"] = torch.bfloat16
    elif args.flux_model_precision == "fp16":
        pipeline_args["torch_dtype"] = torch.float16
    elif args.flux_model_precision == "fp8":
        # assert args.flux_transformer_fp8_path is not None
        assert os.path.exists(args.flux_transformer_fp8_path)
        _TXX = time.perf_counter()
        _transformer = FluxTransformer2DModel.from_single_file(
            args.flux_transformer_fp8_path,
            torch_dtype=torch.bfloat16).to("cuda")
        quantize(_transformer, weights=qfloat8)
        freeze(_transformer)
        pipeline_args["transformer"] = _transformer
        print(f" -- Load fp8 transformer: {args.flux_transformer_fp8_path}, time cost: {time.perf_counter() - _TXX:.03f} sec")
    else:
        raise ValueError("Unreachable branch: unknown precision type ... ")
    if args.enable_cfg_pipeline:
        pipeline_cls = FluxCFGPipeline
        # assert args.negative_prompt is not None
        # assert args.true_cfg > 1
        # assert args.guidance_scale == 1
        print(" -- Switch to flux+cfg pipeline")
    else:
        pipeline_cls = FluxPipeline
        print(" -- Switch to regular flux pipeline (w/o cfg)")

    pipe = pipeline_cls.from_pretrained(**pipeline_args).to("cuda")
    print(f" -- Init pipeline: {time.perf_counter() - _TIC:.03f} sec")

    # Depending on the variant being used, the pipeline call will slightly vary.
    # Refer to the pipeline documentation for more details.
    for i, prompt in enumerate(args.prompts):
        print(f"    {prompt}")
    num_test = len(args.prompts)

    # Visualization utilities
    if args.save_plot:
        ratio_px2mat = float(10 / 1000)  # 1000px => 10unit of matplotlib
        grid_w = float(ratio_px2mat * args.width)
        grid_h = float(ratio_px2mat * args.height)
        if args.plot_orient == "horizontal":
            # horizontal
            fig, ax = plt.subplots(
                args.num_images_per_prompt, num_test,
                figsize=(num_test * grid_w, args.num_images_per_prompt * grid_h))
        elif args.plot_orient == "vertical":
            # vertical
            fig, ax = plt.subplots(
                num_test, args.num_images_per_prompt,
                figsize=(args.num_images_per_prompt * grid_w, num_test * grid_h))
        else:
            raise ValueError("Unreachable branch!")
        if num_test == 1:
            ax = [ax]

    # Enable LoRA (format from diffusers trained)
    _TIC = time.perf_counter()
    prompt_prefix_list =[]
    adapter_names = []
    adapter_weights = []

    if isinstance(args.trigger, str):
        args.trigger = args.trigger.strip()
        if args.trigger != "":
            prompt_prefix_list.append(args.trigger)
    if args.lora_path is not None:
        # Load character-lora:
        adapter_names.append("lora_1")
        adapter_weights.append(args.lora_scale)
        pipe.load_lora_weights(args.lora_path, adapter_name=adapter_names[0])

    if isinstance(args.trigger_2, str):
        args.trigger_2 = args.trigger_2.strip()
        if args.trigger_2 != "":
            prompt_prefix_list.append(args.trigger_2)
    if args.lora_path_2 is not None:
        # Load style-lora:
        adapter_names.append("lora_2")
        adapter_weights.append(args.lora_scale_2)
        pipe.load_lora_weights(args.lora_path_2, adapter_name=adapter_names[1])
    print(f" -- Set LoRA(s) into pipeline: {time.perf_counter() - _TIC:.03f} sec")

    if isinstance(args.extra_triggers, str):
        args.extra_triggers = args.extra_triggers.strip()
        if args.extra_triggers != "":
            prompt_prefix_list.append(args.extra_triggers)

    prompt_prefix = ", ".join(prompt_prefix_list)
    print(f" -- Prompt prefix: {prompt_prefix}")

    # Inference
    os.makedirs(args.out_dir, exist_ok=True)
    out_name_prefix = f"{prompt_prefix}--seed{args.seed}"
    if args.lora_path is not None:
        out_name_prefix += f"--scale_c{args.lora_scale:.02f}"
    if args.lora_path_2 is not None:
        out_name_prefix += f"_s{args.lora_scale_2:.02f}"

    if args.save_plot:
        out_name_fig = f"{out_name_prefix}.jpg" #.png"
        out_file = os.path.join(args.out_dir, out_name_fig)

    for i, prompt in zip(args.prompts_ids, args.prompts):
        set_all_seed(args.seed)
        generator = get_generator(args.seed, "cuda")

        # Reset adapter(s)
        switch_callback = None
        if args.switch_step is None:
            if len(adapter_names) > 0:
                pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
        else:
            if len(adapter_names) > 0:
                pipe.set_adapters(adapter_names[0], adapter_weights[0])
                switch_callback = make_callback(args.switch_step, adapter_names, adapter_weights)
            else:
                print("Warning: if there is no adapters, no need to switch between them")

        # Sample image(s)
        if prompt_prefix != "":
            input_prompt = f"{prompt_prefix}, {prompt}"
        else:
            input_prompt = prompt
        print(input_prompt)
        if args.enable_cfg_pipeline:
            images = pipe(
                prompt=input_prompt,
                negative_prompt=args.negative_prompt,
                true_cfg=args.true_cfg,
                step_start_cfg=args.step_start_cfg,
                height=args.height, width=args.width,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=args.num_images_per_prompt,
                guidance_scale=args.guidance_scale,
                generator=generator,
                joint_attention_kwargs={
                    "scale": 1.0,
                },
                callback_on_step_end=switch_callback).images
        else:
            images = pipe(
                prompt=input_prompt,
                height=args.height, width=args.width,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=args.num_images_per_prompt,
                guidance_scale=args.guidance_scale,
                generator=generator,
                joint_attention_kwargs={
                    "scale": 1.0,
                },
                callback_on_step_end=switch_callback).images
        if args.num_images_per_prompt > 1:
            # separately save each image ...
            for j, img in enumerate(images):
                out_name_img = f"{out_name_prefix}_prompt{i}_sample{j}.jpg" #.png"
                img.save(os.path.join(args.out_dir, out_name_img))
            continue

        img = images[0]
        if args.save_plot:
            ax[i].imshow(img)
            ax[i].axis('off')  # hide axes

        if args.separate_save:
            out_name_img = f"{out_name_prefix}_prompt{i}.jpg" #.png"
            img.save(os.path.join(args.out_dir, out_name_img))

    # Unload LoRA
    pipe.unload_lora_weights()

    # Save output
    if args.save_plot:
        plt.tight_layout()
        fig.savefig(out_file, bbox_inches='tight', pad_inches=0.2)


if __name__ == '__main__':
    args = parse_args()
    # # use FluxCFGPipeline or FluxPipeline
    # args.enable_cfg_pipeline = True
    main(args)
