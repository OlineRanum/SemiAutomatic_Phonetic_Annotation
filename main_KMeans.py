from src.k_means import K_Means
from src.utils.populate_metadata import populate_dataset
import argparse 
import os 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--mode', type=str, help='Set mode for annotating or visualizing', choices=['annotate', 'visualize'], required=True)
    parser.add_argument('--gloss', type=str, help='Gloss for visualization', default='zero')
    parser.add_argument('--k', type=int, help='Number of clusters', default=10)
    parser.add_argument('--input_metadata_path', type=str, help='Path to input metadata', default='data/wlasl_subset.json')
    parser.add_argument('--kmeans_metadata_path', type=str, help='Path to KMeans metadata JSON file', default='data/wlasl_kmeans.json')
    parser.add_argument('--new_labels_path', type=str, help='Path to save KMeans labels text file', default='output/kmeans_labels.txt')

    args = parser.parse_args()

    annotator = K_Means()

    if args.mode == 'annotate':
        annotator.annotate()
        data = annotator.read_KMeans_dict(filename='output/KMeans_dict.txt')
        df = annotator.KMeans_estimation(data, k=args.k)
        annotator.plot_KMeans_scatter(df)
        annotator.output_txt(df['gloss'], df['new_labels'], filename=args.new_labels_path)
        populate_dataset(args.input_metadata_path, args.kmeans_metadata_path, args.new_labels_path)

    elif os.path.exists(args.new_labels_path):
        data = annotator.read_KMeans_dict(filename='output/KMeans_dict.txt')
        df = annotator.KMeans_estimation(data, k=args.k)
        annotator.plot_KMeans_scatter(df)
        annotator.output_txt(df['gloss'], df['new_labels'], filename=args.new_labels_path)
        
    else:
        print('No annotations found, running annotations first.')
        annotator.annotate()
        data = annotator.read_KMeans_dict(filename='output/KMeans_dict.txt')
        df = annotator.KMeans_estimation(data, k=args.k)
        annotator.plot_KMeans_scatter(df)
        annotator.output_txt(df['gloss'], df['new_labels'], filename=args.new_labels_path)