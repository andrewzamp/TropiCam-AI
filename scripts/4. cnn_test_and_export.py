# Libraries
import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model 
from tqdm import tqdm

def classify_images(test_dir, model_path, taxonomy_path, output_csv, confidence_threshold=0.75):
    """
    Classify images using the taxonomic aggregation strategy.
    
    Args:
        test_dir: Directory containing images to classify
        model_path: Path to the trained model
        taxonomy_path: Path to taxonomy mapping CSV
        output_csv: Path for output CSV file
        confidence_threshold: Minimum confidence threshold for taxonomic aggregation
    """
    
    # Load model and taxonomy
    model = load_model(model_path)
    print(f"Model trained on {model.output_shape[-1]} classes")
    taxonomy_df = pd.read_csv(taxonomy_path)
    class_names = taxonomy_df['species'].tolist()
    taxonomic_levels = ['species', 'genus', 'family', 'order', 'class']

    # Get image files recursively
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                image_files.append(os.path.join(root, file))
    image_files.sort()
    
    print(f"Found {len(image_files)} images to process.")

    # Process images and aggregate predictions
    results = []
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load and predict
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.keras.applications.convnext.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array, verbose=0)[0]
            
            # Initialize result
            result = {
                'root_folder': test_dir,
                'relative_path': os.path.relpath(os.path.dirname(img_path), test_dir),
                'file_name': os.path.basename(img_path)
            }
            
            # Aggregate at each taxonomic level
            for level in taxonomic_levels:
                unique_labels = sorted(taxonomy_df[level].unique())
                aggregated = np.zeros(len(unique_labels))
                
                for _, row in taxonomy_df.iterrows():
                    species_idx = class_names.index(row['species'])
                    level_idx = unique_labels.index(row[level])
                    aggregated[level_idx] += prediction[species_idx]
                
                pred_idx = np.argmax(aggregated)
                confidence = np.clip(aggregated[pred_idx], 0.0, 1.0)
                
                result[f'pred_{level}'] = unique_labels[pred_idx]
                result[f'conf_{level}'] = confidence
            
            # Final prediction with configurable threshold
            for level in taxonomic_levels:
                if result[f'conf_{level}'] >= confidence_threshold:
                    result['best_prediction'] = result[f'pred_{level}']
                    result['best_confidence'] = result[f'conf_{level}']
                    result['taxo_level'] = level
                    break
            else:
                result['best_prediction'] = 'Uncertain'
                result['best_confidence'] = 0.0
                result['taxo_level'] = 'None'
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Save results
    if results:
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f'Results saved to {output_csv}')
    else:
        print("No results to save")

# Example usage
if __name__ == "__main__":
    # Configuration
    test_dir = '~/test_images'
    model_path = '~/model/TropiCam_AI_unfrozen'
    taxonomy_path = '~/data/taxonomy_list_mapping.csv'
    output_csv = '~/results/classification_results.csv'
    confidence_threshold = 0.75
    
    # Run classification
    classify_images(test_dir, model_path, taxonomy_path, output_csv, confidence_threshold)