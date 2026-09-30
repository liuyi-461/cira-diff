import sys
from scripts.check_dataset import main
if __name__ == "__main__":
    main(sys.argv[1:] + ["--kind", "latent"])
