import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import glob
from multiprocessing import Process
from tqdm import tqdm
import numpy as np
import re

def get_grid_dimensions(image):
    """Get the grid dimensions from the image"""
    img_w, img_h = image.size
    single_w = img_w // 6  # Assuming 6 columns
    single_h = img_h // 4  # Assuming 4 rows
    return 4, 6, single_w, single_h

def get_caption_template(mode="multi_view"):
    """Return caption template based on mode"""
    templates = {
        "multi_view": "A 24-frame sequence arranged in a 4x6 grid. Each frame captures a 3D model of {subject} from a different angle, rotating 360 degrees. The sequence begins with a front view and progresses through a complete clockwise rotation",
        
        "time_series": "Multi-frame time series captured, arranged as 4x6 grid layout, 24 sequential frames with {d}-frame intervals, individual portrait-oriented frames, composited left-to-right top-to-bottom",
        
        "action_sequence": "24-frame action sequence in 4x6 grid showing {subject} in motion, captured chronologically from left to right, top to bottom, highlighting key moments of movement and transition"
    }
    return templates.get(mode, templates["multi_view"])

def get_subject_content(img_path):
    """Extract subject content from filename or predefined mapping"""
    # Example mapping - extend as needed
    subject_map = {
        "tree": "a tree whose trunk is a twisting pagoda with branches of miniature traditional buildings and roof tile leaves",
        "building": "an architectural model transitioning from traditional to modern design elements",
        "landscape": "a dynamic landscape featuring seasonal transitions across mountain terrain"
    }
    
    # Extract subject from filename
    filename = os.path.basename(img_path)
    for key in subject_map:
        if key in filename.lower():
            return subject_map[key]
    return "the subject" # Default fallback

def process_subset(image_paths, process_id, model_path, mode="multi_view"):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(process_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    base_query = '''Please analyze this image with professional photography descriptions and generate English descriptions.
    Requirements:
    - Describe main subject content, visual features, lighting and atmosphere
    - Use professional photography terminology
    - Generate single line text without breaks
    - Avoid subjective evaluation and repetition
    '''

    for img_path in tqdm(image_paths, desc=f'Process {process_id}'):
        try:
            image = Image.open(img_path).convert('RGB')
            rows, cols, _, _ = get_grid_dimensions(image)
            
            subject = get_subject_content(img_path)
            template = get_caption_template(mode)
            
            if mode == "time_series":
                d = re.search(r'-(\d+)-', os.path.basename(img_path))
                d = d.group(1) if d else "unknown"
                template = template.format(d=d)
            else:
                template = template.format(subject=subject)

            inputs = tokenizer.apply_chat_template([{
                "role": "user", 
                "image": image, 
                "content": base_query
            }], add_generation_prompt=True, tokenize=True, return_tensors="pt")
            
            inputs = inputs.to(device)

            gen_kwargs = {"max_length": 225, "do_sample": True, "top_k": 1}
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                caption = tokenizer.decode(outputs[0])

            caption = caption.replace('<|endoftext|>', '').strip()
            if caption.endswith(','):
                caption = caption[:-1]

            caption += f". {template}"

            txt_path = os.path.splitext(img_path)[0] + '.txt'
            with open(txt_path, 'w') as f:
                f.write(caption)

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

def main():
    MODEL_PATH = "/path/to/glm" # put your glm-4v-9b weights here
    IMAGE_DIR = ".source/vidconcat" # your final data directory
    
    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    
    num_processes = 3
    gpu_ids = [0, 1, 2]
    splits = np.array_split(image_paths, num_processes)

    processes = []
    for i in range(num_processes):
        p = Process(target=process_subset,
                   args=(splits[i].tolist(), gpu_ids[i], MODEL_PATH, "multi_view"))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
