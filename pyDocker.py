#!/usr/bin/env python3
import time
import re
import subprocess
import docker
from docker.errors import APIError, ContainerError, ImageNotFound
import argparse

# Get your machine's public IP
try:
    public_ip = subprocess.check_output(
        ["curl", "--silent", "ifconfig.me"], text=True, timeout=10
    ).strip()
except Exception as e:
    print(f"Warning: could not fetch public IP ({e}), defaulting to localhost")
    public_ip = "127.0.0.1"

def main():
    parser = argparse.ArgumentParser(description="Docker container creation and GPU mapping script.")

    # Add arguments
    parser.add_argument("--base-name", type=str, help="Base name of the Docker container.")
    parser.add_argument("--image", type=str, help="Image to run.")
    parser.add_argument("--data-path", type=str, help="Path to local data folder that will map to /data in container.")
    parser.add_argument("--env-vars", type=str, help="Comma seperated list of extra env vars")
    parser.add_argument("--jupyter-port", type=int, default=5000, help="Base port for Jupyter")
    parser.add_argument("--other-port", type=int, default=8000, help="Base port for other service, such as vLLM server")

    # Parse the arguments
    args = parser.parse_args()

    launch_containers(args)

def parse_env_vars(env_vars_str):
    try:
        return dict(item.split("=", 1) for item in env_vars_str.split(","))
    except ValueError:
        print("Error: Invalid format. Ensure each pair is in the format KEY=VALUE.")
        return {}

def launch_containers(args):
    client = docker.from_env()
    render_ids      = [128, 136, 144, 152, 160, 168, 176, 184]
    base_jupyter    = args.jupyter_port
    base_vllm       = args.other_port
    image           = args.image
    model_volume    = {args.data_path: {"bind": "/data", "mode": "rw"}}
    common_devices  = [
        {"PathOnHost": "/dev/kfd", "PathInContainer": "/dev/kfd", "CgroupPermissions": "rwm"}
    ]
    url_list = []

    for idx, rid in enumerate(render_ids):
        name         = f"{args.base_name}_dev_{rid}"
        jupyter_port = base_jupyter + 50 * idx
        vllm_port    = base_vllm    + 50 * idx

        devices = common_devices + [{
            "PathOnHost":      f"/dev/dri/renderD{rid}",
            "PathInContainer": f"/dev/dri/renderD{rid}",
            "CgroupPermissions": "rwm"
        }]

        env_vars = {"VLLM_PORT": str(vllm_port)}
        if args.env_vars is not None:
            extra_vars = parse_env_vars(args.env_vars)
            env_vars = {**env_vars, **extra_vars}

        try:
            print(f"[{name}] launching container…")
            container = client.containers.run(
                image=image,
                name=name,
                detach=True,
                tty=True,
                devices=devices,
                volumes=model_volume,
                environment=env_vars,
                group_add=["video"],
                security_opt=["seccomp=unconfined"],
                ipc_mode="host",
                network_mode="host",
                cap_add=["SYS_PTRACE"],
                working_dir="/workspace"
            )

            time.sleep(5)

            # Install JupyterLab
            print(f"[{name}] installing jupyterlab…")
            exit_code, install_out = container.exec_run(
                ["python3", "-m", "pip", "install", "--quiet", "jupyterlab"],
                stdout=True, stderr=True
            )
            print(install_out.decode().strip())

            # Start JupyterLab with BSD shell setting for terminals
            print(f"[{name}] starting jupyter-lab on port {jupyter_port}…")
            terminado_flag = "--NotebookApp.terminado_settings='{\"shell_command\": [\"bash\",\"-l\"]}'"
            cmd = (
                f"nohup python3 -m jupyterlab "
                f"--ip=0.0.0.0 --port={jupyter_port} --allow-root "
                f"{terminado_flag} "
                f"> /workspace/jupyter-{jupyter_port}.log 2>&1 &"
            )
            container.exec_run(
                ["bash", "-c", cmd],
                workdir="/workspace",
                detach=True
            )

            # Wait for token to appear
            for _ in range(3):
                time.sleep(5)
                exit_code, list_out = container.exec_run(
                    ["python3", "-m", "jupyter", "lab", "list"],
                    stdout=True, stderr=True
                )
                out = list_out.decode().strip()
                if out:
                    m = re.search(r"token=([^&\s]+)", out)
                    if m:
                        token = m.group(1)
                        url = f"http://{public_ip}:{jupyter_port}/?token={token}"
                        print(f"[{name}] Jupyter URL: {url}")
                        url_list.append(url)
                        print(f"[{name}] VLLM_PORT inside container: {vllm_port}\n")
                    else:
                        print(f"[{name}] could not parse token from: {out}\n")
                    break
            else:
                print(f"[{name}] no token yet, dumping log tail:")
                _, tail_out = container.exec_run(
                    ["bash", "-c", f"tail -n 20 /workspace/jupyter-{jupyter_port}.log"],
                    stdout=True, stderr=True
                )
                print(tail_out.decode().strip(), "\n")

        except ImageNotFound:
            print(f"ERROR: image '{image}' not found — pull or build it first.")
        except ContainerError as e:
            print(f"ERROR: container '{name}' failed:\n{e.stderr.decode()}")
        except APIError as e:
            print(f"ERROR: Docker API error for '{name}':\n{e.explanation}")

    print("All done.\nGenerated URLs:")
    for u in url_list:
        print("  ", u)

if __name__ == "__main__":
    main()
