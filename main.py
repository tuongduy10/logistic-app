"""
Entry point — keep this file minimal.
"""
import warnings
warnings.filterwarnings("ignore")

from pipeline import LogisticPipeline


if __name__ == "__main__":
    LogisticPipeline().run()
