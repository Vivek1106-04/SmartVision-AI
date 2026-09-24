import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Convert Keras .h5 model to TensorFlow.js format")
    parser.add_argument('--model_path', type=str, default='models/exercise_classifier.h5')
    parser.add_argument('--output_dir', type=str, default='public/models/exercise_classifier')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # First, re-save as .keras format (native Keras v3) which the converter handles better
    print(f"Loading model from {args.model_path}...")
    
    # Use tensorflow directly to load and re-save
    import tensorflow as tf
    model = tf.keras.models.load_model(args.model_path)
    
    # Save as SavedModel format (most compatible with tfjs converter)
    saved_model_dir = args.model_path.replace('.h5', '_savedmodel')
    model.export(saved_model_dir)
    print(f"Exported SavedModel to {saved_model_dir}")

    # Use the CLI converter via subprocess to avoid Python import issues
    print(f"Converting to TF.js format in {args.output_dir}...")
    cmd = [
        sys.executable, "-m", "tensorflowjs.converters.converter",
        "--input_format=tf_saved_model",
        "--output_format=tfjs_graph_model",
        saved_model_dir,
        args.output_dir
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"✓ Conversion complete! TF.js model saved to {args.output_dir}")
            # List output files
            for f in os.listdir(args.output_dir):
                fpath = os.path.join(args.output_dir, f)
                size_kb = os.path.getsize(fpath) / 1024
                print(f"  {f} ({size_kb:.1f} KB)")
        else:
            print(f"CLI converter failed: {result.stderr}")
            print("\nFalling back to manual JSON export...")
            fallback_export(model, args.output_dir)
    except Exception as e:
        print(f"CLI converter error: {e}")
        print("\nFalling back to manual JSON export...")
        fallback_export(model, args.output_dir)


def fallback_export(model, output_dir):
    """If tensorflowjs converter fails, export model architecture + weights manually.
    The weights can be loaded in the browser with tf.js loadLayersModel."""
    import json
    import numpy as np
    
    # Save model config as JSON
    config = model.get_config()
    model_json = {
        "modelTopology": {
            "class_name": model.__class__.__name__,
            "config": config,
        },
        "format": "layers-model",
        "generatedBy": "SmartVision-AI pipeline",
        "convertedBy": "manual-export",
        "weightsManifest": [{
            "paths": ["weights.bin"],
            "weights": []
        }]
    }
    
    # Extract weights info and binary. Read the name and the array off the same
    # Variable object: layer.weights and layer.get_weights() are not guaranteed
    # to be in the same order, and zipping them silently mislabels
    # recurrent_kernel as bias. src/core/dlClassifier.js looks the weights up by
    # these names, so a wrong label there breaks inference in the browser.
    weight_data = bytearray()
    for layer in model.layers:
        for variable in layer.weights:
            value = np.asarray(variable)
            weight_data.extend(value.flatten().astype(np.float32).tobytes())
            model_json["weightsManifest"][0]["weights"].append({
                "name": f"{layer.name}/{variable.name}",
                "shape": list(value.shape),
                "dtype": "float32"
            })

    # Write files
    json_path = os.path.join(output_dir, "model.json")
    with open(json_path, "w") as f:
        json.dump(model_json, f, indent=2, default=str)
    
    bin_path = os.path.join(output_dir, "weights.bin")
    with open(bin_path, "wb") as f:
        f.write(weight_data)
    
    total_kb = (os.path.getsize(json_path) + os.path.getsize(bin_path)) / 1024
    print(f"✓ Fallback export complete! Files saved to {output_dir}")
    print(f"  model.json ({os.path.getsize(json_path)/1024:.1f} KB)")
    print(f"  weights.bin ({os.path.getsize(bin_path)/1024:.1f} KB)")
    print(f"  Total: {total_kb:.1f} KB")


if __name__ == '__main__':
    main()
