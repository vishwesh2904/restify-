import argparse
from generate_synthetic_data import generate_insomnia_synthetic_data

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic insomnia data with custom sample size.")
    parser.add_argument(
        "-n", "--num_samples", type=int, default=2000,
        help="Number of synthetic samples to generate (default: 2000)"
    )
    args = parser.parse_args()
    generate_insomnia_synthetic_data(n_samples=args.num_samples)

if __name__ == "__main__":
    main()
