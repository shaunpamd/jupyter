#!/usr/bin/env python3
import time
import re
import subprocess
import docker
from docker.errors import APIError, ContainerError, ImageNotFound

# Get your machine's public IP
try:
    public_ip = subprocess.check_output(
        ["curl", "--silent", "ifconfig.me"], text=True, timeout=10
    ).strip()
except Exception as e:
    print(f"Warning: could not fetch public IP ({e}), defaulting to localhost")
    public_ip = "127.0.0.1"

def launch_containers():
    client = docker.from_env()
    render_ids      = [128, 136, 144, 152, 160, 168, 176, 184]
    base_jupyter    = 5000
    base_vllm       = 8000
    image           = "rocm/dev-ubuntu-22.04:6.3.4-complete"
    model_volume    = {"/mnt/data": {"bind": "/data", "mode": "rw"}}
    common_devices  = [
        {"PathOnHost": "/dev/kfd", "PathInContainer": "/dev/kfd", "CgroupPermissions": "rwm"}
    ]
    url_list = []

    for idx, rid in enumerate(render_ids):
        name         = f"vllm_dev_{rid}"
        jupyter_port = base_jupyter + 50 * idx
        vllm_port    = base_vllm    + 50 * idx

        devices = common_devices + [{
            "PathOnHost":      f"/dev/dri/renderD{rid}",
            "PathInContainer": f"/dev/dri/renderD{rid}",
            "CgroupPermissions": "rwm"
        }]

        env_vars = {"VLLM_PORT": str(vllm_port), "HF_HUB_CACHE": "/data/huggingface/hub"}

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
    launch_containers()
