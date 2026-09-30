"""Single-sample ensemble forecast and visualization, with explicit data provenance."""
import sys
from scripts.evaluate import main as evaluate

def main():
    evaluate(sys.argv[1:] + ['--max_samples','1','--figure'])

if __name__ == '__main__':
    main()
