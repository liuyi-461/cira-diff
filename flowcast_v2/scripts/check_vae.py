"""VAE quality metrics and visualization on an independent GOES split."""
import sys
from scripts.evaluate import main as evaluate

def main():
    evaluate(sys.argv[1:] + ['--figure'])

if __name__ == '__main__':
    main()
