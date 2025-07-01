ec2_helper.py – zero-friction helper suite for EC2s

Prerequisites
-------------
1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Make the script executable: `chmod +x ec2_helper.py`
3. Create .env file with AWS credentials:
   ```
   AWS_ACCESS_KEY_ID=your_key_here
   AWS_SECRET_ACCESS_KEY=your_secret_here
   ```
4. Ensure the IAM role "EC2HelperSSMProfile" exists with SSM permissions

Basic CLI examples
------------------
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

The script:
  • Defaults to us-east-1 unless you pass --region
  • Identifies "your" instance by the tag value you supply via --tag
  • Uses AWS SSM for everything (no SSH keys / security-group headaches)
  • Auto-termination enforced via uptime checks during polling operations
  • Reads AWS creds from .env (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
  • Supports standard AL2023, GPU-optimized, and Deep Learning AMIs
  • Includes file transfer via S3, volume resizing, and polling utilities

Features
--------
- **Instance Management**: Create, find, or reuse EC2 instances with smart tagging
- **AMI Options**: Standard Amazon Linux 2023, GPU-optimized (NVIDIA), or Deep Learning AMIs
- **Volume Management**: Automatic EBS volume resizing with filesystem extension
- **Remote Execution**: Run commands via AWS SSM (no SSH required)
- **File Transfer**: Upload/download files using S3 as intermediary
- **Polling**: Wait for commands to succeed or monitor background processes
- **Auto-termination**: Uptime-based termination during polling operations prevents runaway costs

Importing
---------
>>> import ec2_helper as ec2
>>> inst, was_created = ec2.spin_up_or_find("g5.xlarge", tag="mybox", dlami=True)
>>> ec2.run_command(inst.id, ["python", "-c", "print('hi')"])
>>> ec2.upload_file(inst.id, pathlib.Path("model.pkl"))
>>> ec2.poll_until_command_succeeds(inst.id, "systemctl status nginx")