import argparse, torch, json, os
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation",         type = str,     default = "/project/g/r13922043/hw3_data/p2_data/val.json")
    parser.add_argument("--pred_file",          type = str,     default = "/project/g/r13922043/hw3_output/P2_pred")
    return parser.parse_args()

def main():
    config = parse()
    with open(config.annotation) as f:
        annotation = json.load(f)
    with open(config.pred_file) as f:
        predictions = json.load(f)
    dicts = {}
    for item in annotation["images"]:
        dicts[item["file_name"]] = item["id"]
    updated_predictions = {}
    for file_name, prediction in predictions.items():
        #image_id = dicts[file_name]  # Ensure image_id matches the type in dicts
        if file_name:
            # remove the file extension
            image_id = file_name.split(".")[0]
            updated_predictions[image_id] = prediction
        else:
            print(f"Warning: image_id {image_id} not found in annotations")

    directory, filename = os.path.split(config.pred_file)
    name, ext = os.path.splitext(filename)
    new_filename = f"update_{name}{ext}"
    new_path = os.path.join(directory, new_filename)

    print(f"File renamed to: {new_path}")

    with open(new_path, "w") as f:
        f.write(json.dumps(updated_predictions, indent=4))

if __name__ == '__main__':
    torch.manual_seed(42)
    main()