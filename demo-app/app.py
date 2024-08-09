import time
from threading import Thread
import gradio as gr
import torch
from PIL import Image
import cv2
from PIL import Image
from ultralytics import YOLO
import httpcore
setattr(httpcore, 'SyncHTTPTransport', 'AsyncHTTPProxy')
from googletrans import Translator
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
import os


yolo = YOLO('yolov8n.pt')

model_id = "vikhyatk/moondream2"
revision = "308043be540ca5c09d369bc96dd96991131f9135"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
dtype = None 
load_in_4bit = True
llm, llm_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "NexusAI-tddi/Qwen2-72B-Instruct-OpenOrca-tr", 
    max_seq_length = 8192,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)


tmp_cache = {}

PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
   <img src="https://cdn-uploads.huggingface.co/production/uploads/6468ce47e134d050a58aa89c/Gm9ALxLKmM1mJXRHfwTEg.png" style="width: 40%; height: auto; opacity: 0.55;  "> 
   <h1 style="font-size: 24px; margin-bottom: 2px; opacity: 0.65;">🚀👁️ Görme Yeteneğine Sahip Türkçe bir Chatbot!</h1>
   <p style="font-size: 16px; margin-bottom: 2px; opacity: 0.75;">🚀👁️ Görme Yeteneğine Sahip Türkçe bir Chatbot!<br>🖼️ Bir resim yükleyin ve istediğiniz her şeyi sorun!</p>
</div>
"""


def gr_history_to_openai_history(history):
    return [
        {"role": role, "content": content}
        for entry in history if entry[1] is not None
        for role, content in zip(["user", "assistant"], entry)
    ]


def get_position_label(x_center, y_center, img_width, img_height):
    horizontal_position = 'left' if x_center < img_width / 3 else 'right' if x_center > 2 * img_width / 3 else 'center'
    vertical_position = 'top' if y_center < img_height / 3 else 'bottom' if y_center > 2 * img_height / 3 else 'middle'
    return f'{vertical_position} {horizontal_position}'


def translate_en(text):
    translator = Translator()
    return {translator.translate(text, dest="en").text}


def vision_prompt(image, question, language):
    global model, tmp_cache
    if image in list(tmp_cache.keys()):
        img = cv2.imread(image)
        return tmp_cache[image], img
    
    model = model.to("cuda")

    translator = Translator()
    results = yolo(image)
    img = cv2.imread(image)
    img_height, img_width, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    object_counter = {}
    prompt = []
    img_pil = Image.fromarray(img)
    enc_image = model.encode_image(img_pil).to("cuda")

    ##### ask moondream for descirption and the question itself
    
    description = model.answer_question(enc_image, "Describe this image in detailed language.", tokenizer)
    description = translator.translate(description, dest=language).text # translate
    
    answer = model.answer_question(enc_image, translate_en(question), tokenizer, pad_token_id=tokenizer.eos_token_id)
    answer = translator.translate(answer, dest=language).text # translate

    #####

    prompt.append(f"Bütün görselin açıklaması: {description}")

    max_object_count = 4
    
    for i, box in enumerate(results[0].boxes if len(results[0].boxes)<max_object_count else results[0].boxes[:max_object_count]): # max 4 object
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        class_id = results[0].names[box.cls[0].item()]

        if class_id not in object_counter:
            object_counter[class_id] = 0
        else:
            object_counter[class_id] += 1
        cls = translator.translate(class_id, dest=language).text
        indexed_class_id = f"{cls}_{object_counter[class_id]}"
        conf = round(box.conf[0].item(), 2)
        x_center = (cords[0] + cords[2]) / 2
        y_center = (cords[1] + cords[3]) / 2
        cv2.rectangle(img, (cords[0], cords[1]), (cords[2], cords[3]), (255, 0, 0), 2)
        cv2.putText(img, indexed_class_id, (cords[0], cords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        position_label = get_position_label(x_center, y_center, img_width, img_height)
        position_label = translator.translate(position_label, dest=language).text
        crop_img = img[cords[1]:cords[3], cords[0]:cords[2]]
        crop_img_pil = Image.fromarray(crop_img)
        enc_image = model.encode_image(crop_img_pil).to("cuda")
        
        description = model.answer_question(enc_image, "Describe this image in undetailed language.", tokenizer)
        description = translator.translate(description, dest=language).text
        
        prompt.append(f"{translator.translate('Object', dest=language).text}: {indexed_class_id}, {translator.translate('Position', dest=language).text}: {position_label}, {translator.translate('Description', dest=language).text}: {description}")

    
    prompt.append(f"Bu soru için başka bir görebilen model tarafından oluşturulan cevap: {answer}")
    labeled_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    tmp_cache[image] = prompt
    return prompt, labeled_image


def bot_streaming(message, history):
    if message["files"]:
        if type(message["files"][-1]) == dict:
            image = message["files"][-1]["path"]
        else:
            image = message["files"][-1]
    else:
        for hist in history:
            if type(hist[0]) == tuple:
                image = hist[0][0]

    text_message = message['text']

    context = '\n\n'.join(vision_prompt(image, text_message, 'tr')[0])
    
    system = """You are a Language Model that can answer questions about its environment as if it has vision capabilities. The model should respond based on the given context. You should not talk about the existince of other vision models that helps you.

Example Interaction:

User: "What do you see?

You: Respond based on the context provided.

Context:

{}
""".format(context)

    print(system)
    
    messages = [{"role": "system", "content": system}] + gr_history_to_openai_history(history) + [{"role": "user", "content": text_message}]

    text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = llm_tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = llm.generate(
        model_inputs.input_ids,
        max_new_tokens=8192
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    outputs = ""
    for i in generated_ids:
        outputs += llm_tokenizer.decode(i, skip_special_tokens=True)
        yield outputs


chatbot = gr.Chatbot(placeholder=PLACEHOLDER, scale=1)
chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"],
                                  placeholder="Mesajınızı yazın veya resminizi yükleyiniz...", show_label=False)

with gr.Blocks(fill_height=True) as demo:
    gr.ChatInterface(
        fn=bot_streaming,
        title="🚀👁️ Görme Yeteneğine Sahip Türkçe bir Chatbot!",
        description="🚀👁️ Görme Yeteneğine Sahip Türkçe bir Chatbot! Bir resim yükleyin ve istediğiniz her şeyi sorun!",
        stop_btn="Çıktı üretimini durdur",
        multimodal=True,
        textbox=chat_input,
        chatbot=chatbot,
    )

demo.queue(api_open=False)
demo.launch(show_api=False, share=False)
