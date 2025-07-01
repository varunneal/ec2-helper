# EC2 Helper

ec2_helper.py – Single-file helper suite for EC2s

## Prerequisites

1. Install uv (recommended): `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. In your $PATH create an .env file with AWS credentials:
   ```
   AWS_ACCESS_KEY_ID=your_key_here
   AWS_SECRET_ACCESS_KEY=your_secret_here
   ```
3. In AWS create an IAM role "EC2HelperSSMProfile" with SSM and EC2 permissions


## Installation

This is just a single script so you can easily clone it or copy it. However, I reccomend you use [uv](https://github.com/astral-sh/uv).

**Installing as a module:**
You can use pip (`pip install git+https://github.com/varunneal/ec2-helper@main`) or uv:

```
# Install to your environment 
uv pip install git+https://github.com/varunneal/ec2-helper@main

# Add as a dependency to your package
uv add git+https://github.com/varunneal/ec2-helper@main
```

Hint: For scripts, you can import this module directly into your [shebang](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies).

**Running from the CLI:**

Either clone and run

```
chmod +x ec2_helper.py
./ec2_helper.py --help
```

or run the raw script

```
uv run https://raw.githubusercontent.com/varunneal/ec2-helper/refs/heads/main/ec2_helper.py
```


## Importing

```
import ec2_helper as ec2

# dlami is the Deep Learning AMI for GPU-enabled EC2s
inst, was_created = ec2.spin_up_or_find("g5.xlarge", tag="mybox", dlami=True)
ec2.run_command(inst.id, ["python", "-c", "print('hi')"])
ec2.upload_file(inst.id, pathlib.Path("model.pkl"))
ec2.poll_until_command_succeeds(inst.id, "systemctl status nginx")
```

## Basic CLI examples


```
# create or reuse a box (with GPU support)
./ec2_helper.py spin-up-or-find --instance-type g4dn.xlarge --tag whisper-gpu --gpu

# create or reuse a box (with Deep Learning AMI)
./ec2_helper.py spin-up-or-find --instance-type g5.xlarge --tag ml-training --dlami

# run a one-off command
./ec2_helper.py run --instance-id i-0abc123 -- bash "nvidia-smi"

# install uv and resize volume (default 32GB)
./ec2_helper.py setup --instance-id i-0abc123 --volume-size 64

# upload/download files via S3
./ec2_helper.py upload --instance-id i-0abc123 --local-file model.pkl
./ec2_helper.py download --instance-id i-0abc123 --remote-path /tmp/results.json

# poll until command succeeds
./ec2_helper.py poll --instance-id i-0abc123 --command "systemctl status nginx"

# launch background process and monitor
./ec2_helper.py launch-bg --instance-id i-0abc123 --command "python train.py"
./ec2_helper.py poll-bg --instance-id i-0abc123 --pid 12345 --success-condition "test -f /tmp/done"
```


## More info

- **AMI Options**: Standard Amazon Linux 2023, GPU-optimized (NVIDIA), or Deep Learning AMIs
- **Volume Management**: Automatic EBS volume resizing with filesystem extension
- **No SSH**: Run commands via AWS SSM. Upload/download files using S3 as intermediary (cba to handle ssh keys)
- **Polling**: Wait for commands to succeed or monitor background processes

