#!/usr/bin/env python3
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Docker container creation and GPU mapping script.")
    parser.add_argument("--base-name", type=str, help="Base name of the Docker container.")
    parser.add_argument("--count", type=int, default=8, help="Base port for other service, such as vLLM server")
    args = parser.parse_args()

    cleanup_containers(args)

def cleanup_containers(args):
    # The same IDs used when launching
    render_ids = [128, 136, 144, 152, 160, 168, 176, 184]

    if args.count < 8:
        render_ids = render_ids[:args.count]

    for rid in render_ids:
        name = f"{args.base_name}_lab_{rid}"
        print(f"Stopping container {name}…")
        try:
            subprocess.run(["docker", "stop", name], check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"  ⚠️  Could not stop {name} (it may not be running)")

        print(f"Removing container {name}…")
        try:
            subprocess.run(["docker", "rm", name], check=True, stdout=subprocess.DEVNULL)
            print(f"  🗑  {name} removed")
        except subprocess.CalledProcessError:
            print(f"  ⚠️  Could not remove {name} (it may not exist)")

    print("Cleanup complete.")

if __name__ == "__main__":
    main()
