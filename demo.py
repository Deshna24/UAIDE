from detector import process_image, scan_dataset
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset folder')
    parser.add_argument('--image', type=str, help='single image path')
    parser.add_argument('--out', type=str, default='out')
    args = parser.parse_args()
    if args.image:
        r = process_image(args.image, out_dir=args.out)
        print(f"Score: {r['ai_score']:.4f}")
    elif args.dataset:
        scan_dataset(args.dataset, out_dir=args.out)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
