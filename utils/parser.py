import os

def parse_box_file(file_path):
    """
    Parses a single SROIE box file and extracts the full ground truth text.
    Format: x1,y1,x2,y2,x3,y3,x4,y4,TEXT
    Since TEXT can contain commas, we split by comma with maxsplit=8.
    """
    text_lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',', 8)
                if len(parts) == 9:
                    text_lines.append(parts[8])
                elif len(parts) > 0:
                    pass
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return " ".join(text_lines)

def get_dataset_files(dataset_path):
    """
    Returns a list of dictionaries with image_path and gt_text for the test set.
    """
    test_img_dir = os.path.join(dataset_path, 'test', 'img')
    test_box_dir = os.path.join(dataset_path, 'test', 'box')
    
    data = []
    if not os.path.exists(test_img_dir) or not os.path.exists(test_box_dir):
        print(f"Dataset path invalid or test folders missing: {dataset_path}")
        return data
        
    for filename in os.listdir(test_img_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(test_img_dir, filename)
            base_name = os.path.splitext(filename)[0]
            box_path = os.path.join(test_box_dir, f"{base_name}.txt")
            
            if os.path.exists(box_path):
                gt_text = parse_box_file(box_path)
                data.append({
                    'image_name': filename,
                    'image_path': img_path,
                    'gt_text': gt_text
                })
            else:
                print(f"Warning: Missing box file for image {filename}")
    return data
