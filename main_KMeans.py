from src.k_means import K_Means
import argparse 
import os 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--mode', type=str, help='Set mode for annotating or visualizing', choices=['annotate', 'visualize'], required=True)
    parser.add_argument('--gloss', type=str, help='Gloss for visualization', default='zero')
    parser.add_argument('--k', type=int, help='Number of clusters', default=2)
    
    args = parser.parse_args()

    annotator = K_Means()
    
    if args.mode == 'annotate':
        annotator.annotate()
        data = annotator.read_KMeans_dict(filename='output/KMeans_dict.txt')
        df = annotator.KMeans_estimation(data, k = args.k)
        annotator.plot_KMeans_scatter(df)
        annotator.output_txt(df['gloss'], df['new_labels'], filename='output/kmeans_labels.txt')

    elif os.path.exists('output/kmeans_labels.txt'):  
            data = annotator.read_KMeans_dict(filename='output/KMeans_dict.txt')
            df = annotator.KMeans_estimation(data, k = args.k)
            annotator.plot_KMeans_scatter(df)
            annotator.output_txt(df['gloss'], df['new_labels'], filename='output/kmeans_labels.txt')
    else:
        print('No annotations found, running annotations first.')
        annotator.annotate()
        data = annotator.read_KMeans_dict(filename='output/KMeans_dict.txt')
        df = annotator.KMeans_estimation(data, k = args.k)
        annotator.plot_KMeans_scatter(df)
        annotator.output_txt(df['gloss'], df['new_labels'], filename='output/kmeans_labels.txt')
    