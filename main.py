import os
import argparse
import logging
from dotenv import load_dotenv
from openai import OpenAI

from karma_mini.core import KARMAPipeline

# Load environment variables from .env file
load_dotenv()

# Setup configuration
API_KEY = os.getenv("KIT_API_KEY")
BASE_URL = os.getenv("KIT_BASE_URL")

AVAILABLE_MODELS = [
    "kit.gemma4-31b-it",
    "kit.gpt-oss-120b",
    "kit.minimax-m2.5-229b",
    "kit.minimax-m2.7-229b",
    "kit.mistral-small-4-119b-a8b"
]

def main():
    parser = argparse.ArgumentParser(description="KARMA Mini Pipeline")
    parser.add_argument(
        "--model", 
        choices=AVAILABLE_MODELS, 
        default="kit.mistral-small-4-119b-a8b", # Using mistral as default for better reasoning
        help="Select the LLM model to query"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/abstracts.json",
        help="Path to the JSON file containing abstracts"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0, # Increased timeout for agent tasks
        help="API request timeout in seconds (default: 60.0)"
    )
    args = parser.parse_args()

    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 60)
    print(" KARMA Mini Pipeline ".center(60, "="))
    print("=" * 60)
    print(f"Model    : {args.model}")
    print(f"Data     : {args.data}")
    print(f"Timeout  : {args.timeout}s")
    print("-" * 60)

    if not API_KEY or not BASE_URL:
        print("\n[ERROR] KIT_API_KEY or KIT_BASE_URL environment variable is not set!")
        return

    # Initialize the client
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=args.timeout)

    try:
        pipeline = KARMAPipeline(client, args.model)
        pipeline.process_abstracts(args.data)
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()
