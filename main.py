import os
import argparse
import openai
from dotenv import load_dotenv
from openai import OpenAI

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
    parser = argparse.ArgumentParser(description="KIT Toolbox LLM Hello World Setup")
    parser.add_argument(
        "--model", 
        choices=AVAILABLE_MODELS, 
        default="kit.gemma4-31b-it",
        help="Select the LLM model to query (default: kit.gemma4-31b-it)"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Erkläre das Prinzip der Rayleigh-Streuung in drei Sätzen.",
        help="The query to send to the model"
    )
    parser.add_argument(
        "--system", 
        type=str, 
        default="Du bist ein hilfreicher Assistent am KIT.",
        help="System message for the model"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="API request timeout in seconds (default: 30.0)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" KIT Toolbox LLM Experiment Client ".center(60, "="))
    print("=" * 60)
    print(f"Base URL : {BASE_URL}")
    print(f"Model    : {args.model}")
    print(f"System   : {args.system}")
    print(f"Prompt   : {args.prompt}")
    print(f"Timeout  : {args.timeout}s")
    print("-" * 60)

    if not API_KEY:
        print("\n[ERROR] KIT_API_KEY environment variable is not set!")
        print("Please define it in your '.env' file or set the environment variable directly.")
        print("Example '.env' file content:")
        print("  KIT_API_KEY=your_actual_api_key_here")
        print("=" * 60)
        return

    if not BASE_URL:
        print("\n[ERROR] KIT_BASE_URL environment variable is not set!")
        print("Please define it in your '.env' file or set the environment variable directly.")
        print("Example '.env' file content:")
        print("  KIT_BASE_URL=https://ki-toolbox.scc.kit.edu/api/v1")
        print("=" * 60)
        return

    # Initialize the client with the specified timeout
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=args.timeout)

    try:
        print("Sending request to KIT Toolbox API...")
        chat_completion = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": args.system},
                {"role": "user", "content": args.prompt},
            ],
        )
        print("\n--- Response ---")
        print(chat_completion.choices[0].message.content)
        print("=" * 60)
    except openai.APITimeoutError:
        print(f"\n[TIMEOUT ERROR] The request timed out after {args.timeout} seconds.")
        print("The selected model might be offline, under heavy load, or there is a network issue.")
        print("Please try again, increase the --timeout, or switch to a different model (e.g. --model kit.mistral-small-4-119b-a8b).")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Ein Fehler ist aufgetreten: {e}")
        print("Please check your network connection or API credentials in the .env file.")
        print("=" * 60)

if __name__ == "__main__":
    main()
