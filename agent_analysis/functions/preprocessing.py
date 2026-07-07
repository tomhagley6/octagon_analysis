import os
import yaml

def extract_step_penalties(parent_dir, yaml_filename=None):
    results = []
    step_penalties = []
    thetas = []

    # Ensure consistent ordering of subfolders
    subfolders = sorted([
        f for f in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, f))
    ])

    for folder in subfolders:
        folder_path = os.path.join(parent_dir, folder)

        # Option 1: known YAML filename
        if yaml_filename:
            yaml_path = os.path.join(folder_path, yaml_filename)
        else:
            # Option 2: find first .yaml or .yml file
            yaml_files = [
                f for f in os.listdir(folder_path)
                if f.endswith((".yaml", ".yml"))
            ]
            if not yaml_files:
                print(f"No YAML file in {folder}")
                continue
            yaml_path = os.path.join(folder_path, yaml_files[0])

            #print(f"Processing {yaml_path} for folder {folder}")

        # Load YAML and extract value
        try:
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)

            step_penalty = data.get("environment_parameters", {}).get("step_penalty")
            translation_cost = data.get("environment_parameters", {}).get("translation_cost")
            turning_cost = data.get("environment_parameters", {}).get("turning_cost")
            
            results.append((folder, step_penalty, translation_cost, turning_cost))
            step_penalties.append(step_penalty)
            thetas.append((translation_cost, turning_cost))

        except Exception as e:
            print(f"Error processing {yaml_path}: {e}")

    return results, step_penalties, thetas