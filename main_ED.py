from src.euclidean_distance import EuclideanDistance
from src.utils.populate_metadata import populate_dataset
import argparse 
import os 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script for annotating or visualizing with Euclidean Distance.")
    parser.add_argument('--mode', type=str, help='Set mode for annotating or visualizing', choices=['annotate', 'visualize'], required=True)
    parser.add_argument('--gloss', type=str, help='Gloss for visualization', default='zero')
    parser.add_argument('--input_metadata_path', type=str, help='Path to input metadata', default='data/wlasl_subset.json')
    parser.add_argument('--ed_metadata_path', type=str, help='Path to Euclidean Distance metadata JSON file', default='data/wlasl_ed.json')
    parser.add_argument('--new_labels_path', type=str, help='Path to save Euclidean Distance labels text file', default='output/ED_labels.txt')

    args = parser.parse_args()

    annotator = EuclideanDistance()
    
    if args.mode == 'annotate':
        annotator.annotate()
        annotator.plot_ED_heatmap(args.gloss)
        populate_dataset(args.input_metadata_path, args.ed_metadata_path, args.new_labels_path)

    elif os.path.exists(args.new_labels_path):  
        annotator.plot_ED_heatmap(args.gloss)

    else:
        print('No annotations found, running annotations first.')
        annotator.annotate()
        annotator.plot_ED_heatmap(args.gloss)
        populate_dataset(args.input_metadata_path, args.ed_metadata_path, args.new_labels_path)
        