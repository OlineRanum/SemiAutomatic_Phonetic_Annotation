from src.euclidean_distance import EuclideanDistance
import argparse 
import os 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--mode', type=str, help='Set mode for annotating or visualizing', choices=['annotate', 'visualize'], required=True)
    parser.add_argument('--gloss', type=str, help='Gloss for visualization', default='zero')
    
    args = parser.parse_args()

    annotator = EuclideanDistance()
    
    if args.mode == 'annotate':
        annotator.annotate()
        annotator.plot_ED_heatmap(args.gloss)

    elif os.path.exists('output/ED_labels.txt'):  
        annotator.plot_ED_heatmap(args.gloss)

    else:
        print('No annotations found, running annotations first.')
        annotator.annotate()
        annotator.plot_ED_heatmap(args.gloss)
    