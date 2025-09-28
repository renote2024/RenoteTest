from diskcache import Index
import pandas as pd
import os
import argparse

def main(cache_path, csv_path):
    cache = Index(cache_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    results = [{"nb_path": k, **v} for k, v in cache.items()]
    df = pd.DataFrame(results)
    df.to_csv(csv_path)
    print(f"✅ Results saved to {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export DiskCache index to CSV.')

    parser.add_argument('--cache_path', type=str, required=True,
                        help='Path to the DiskCache index directory')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to save the resulting CSV file')

    args = parser.parse_args()
    main(cache_path=args.cache_path, csv_path=args.csv)
