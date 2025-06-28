#!/usr/bin/env python
# uv python dependencies:
#   - boto3>=1.34
#   - botocore>=1.34
#   - typer>=0.12
#   - python-dotenv>=1.0
#   - rich>=13.7
"""
ec2_helper.py – zero-friction spin-up / find / setup for EC2

Basic CLI examples
------------------
# create or reuse a box (with GPU support)
uv run ec2_helper.py spin-up-or-find --instance-type g4dn.xlarge --tag whisper-gpu --gpu

# create or reuse a box (with Deep Learning AMI)
uv run ec2_helper.py spin-up-or-find --instance-type g5.xlarge --tag ml-training --dlami

# run a one-off command
uv run ec2_helper.py run --instance-id i-0abc123 -- bash "nvidia-smi"

# install uv and resize volume (default 32GB)
uv run ec2_helper.py setup --instance-id i-0abc123 --volume-size 64

# upload/download files via S3
uv run ec2_helper.py upload --instance-id i-0abc123 --local-file model.pkl
uv run ec2_helper.py download --instance-id i-0abc123 --remote-path /tmp/results.json

# poll until command succeeds
uv run ec2_helper.py poll --instance-id i-0abc123 --command "systemctl status nginx"

The script:
  • Defaults to us-east-1 unless you pass --region
  • Identifies “your” instance by the tag value you supply via --tag
  • Uses AWS SSM for everything (no SSH keys / security-group headaches)
  • Auto-termination enforced via uptime checks during polling operations
  • Reads AWS creds from .env (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)

Importing
---------
>>> import ec2_helper as ec2
>>> inst, was_created = ec2.spin_up_or_find("g5.xlarge", tag="mybox")
>>> ec2.run_command(inst.id, ["python", "-c", "print('hi')"])
"""
from __future__ import annotations

import io
import json
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import boto3
import botocore.exceptions
import typer
from dotenv import load_dotenv
from rich import print

# -----------------------------------------------------
#  Essentials
# -----------------------------------------------------
DEFAULT_REGION = "us-east-1"
APP_TAG_KEY = "ec2-helper-tag"  # We tag every instance with this key


def _session(region: Optional[str] = None):
    """Return a boto3.Session after loading .env."""
    load_dotenv()
    return boto3.Session(region_name=region or DEFAULT_REGION)


# -----------------------------------------------------
#  Core helpers
# -----------------------------------------------------
def _latest_al2023_ami(session: boto3.Session, *, gpu: bool = False, dlami: bool = False) -> str:
    """
    Return the newest Amazon Linux 2023 AMI.
    
      gpu=False -> vanilla GA image
      gpu=True  -> ECS GPU-optimised image (NVIDIA 570-series driver, CUDA-12)
      dlami=True -> Deep Learning AMI (GPU, PyTorch family, AL2023)
                   (gpu flag is ignored if dlami=True)
    
    Example:
        ami_id = _latest_al2023_ami(session, dlami=True)
    """
    ec2c = session.client("ec2")
    
    if dlami:
        pattern = "Deep Learning*GPU*PyTorch*Amazon Linux 2023*"
    elif gpu:
        pattern = "al2023-ami-ecs-gpu-hvm-*x86_64-ebs"  # GPU-optimized ECS image
    else:
        pattern = "al2023-ami-2023*-x86_64"  # Standard GA pattern
    
    # Search all public AMIs without specifying owner
    imgs = ec2c.describe_images(
        Filters=[
            {"Name": "name", "Values": [pattern]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "is-public", "Values": ["true"]},  # Only public AMIs
        ],
    )["Images"]
    
    # Filter to Amazon-owned AMIs by checking the owner alias or known patterns
    amazon_imgs = []
    for img in imgs:
        # Amazon AMIs have owner-alias of 'amazon' or are from known Amazon accounts
        if (img.get("ImageOwnerAlias") == "amazon" or 
            img.get("OwnerId") in ["137112412989", "591542846629", "898082745236"] or  # Known Amazon account IDs
            "amazon" in img.get("Description", "").lower() or
            "deep learning ami" in img.get("Description", "").lower()):
            amazon_imgs.append(img)
    
    if not amazon_imgs:
        raise RuntimeError(f"No Amazon-owned AMIs found matching pattern: {pattern}")
    
    latest = max(amazon_imgs, key=lambda x: x["CreationDate"])
    return latest["ImageId"]


def _find_instance(session: boto3.Session, tag: str):
    ec2 = session.resource("ec2")
    insts = list(
        ec2.instances.filter(
            Filters=[
                {"Name": f"tag:{APP_TAG_KEY}", "Values": [tag]},
                {"Name": "instance-state-name", "Values": ["pending", "running"]},
            ]
        )
    )
    return insts[0] if insts else None


def spin_up_instance(
    instance_type: str, tag: str, region: Optional[str] = None, key_name: Optional[str] = None, gpu: bool = False, dlami: bool = False, auto_terminate_hours: float = 3.0
):
    """
    Start a fresh EC2 instance (Amazon Linux 2023) and tag it.
    Auto-termination is enforced via uptime checks during polling operations.
    Returns: (instance, True) - True indicates instance was created.
    """
    sess = _session(region)
    ec2 = sess.resource("ec2")

    ami = _latest_al2023_ami(sess, gpu=gpu, dlami=dlami)
    if dlami:
        ami_type = "Deep Learning AMI"
    elif gpu:
        ami_type = "GPU-optimized"
    else:
        ami_type = "Standard"
    print(f"[bold green]Using AMI[/]: {ami} ({ami_type})")

    inst = ec2.create_instances(
        ImageId=ami,
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1,
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": APP_TAG_KEY, "Value": tag},
                    {"Key": "Name", "Value": f"{tag}-{instance_type}"},
                ],
            }
        ],
        IamInstanceProfile={"Name": "EC2HelperSSMProfile"},  # needs to exist
    )[0]

    print(f"∙ Waiting for instance {inst.id} to be running…")
    inst.wait_until_running()
    inst.reload()

    print(
        f"[bold cyan]Instance ready:[/] {inst.id} "
        f"({inst.instance_type}) in {inst.placement['AvailabilityZone']}"
    )
    return inst, True










def spin_up_or_find(
    instance_type: str,
    tag: str,
    region: Optional[str] = None,
    gpu: bool = False,
    dlami: bool = False,
    auto_terminate_hours: float = 3.0,
):
    """
    Find a running instance with the given tag, otherwise create one.
    Auto-termination is enforced via uptime checks during polling operations.
    Returns: (instance, was_created) - was_created is True if new instance was created, False if existing found.
    """
    sess = _session(region)
    inst = _find_instance(sess, tag)
    if inst:
        print(f"[bold yellow]Reusing[/] {inst.id}")
        return inst, False
    return spin_up_instance(instance_type, tag, region, key_name=None, gpu=gpu, dlami=dlami, auto_terminate_hours=auto_terminate_hours)


# -----------------------------------------------------
#  SSM utilities
# -----------------------------------------------------
def _ssm_send(
    session: boto3.Session,
    instance_id: str,
    commands: List[str],
    comment: str = "",
    timeout: int = 3600,
):
    ssm = session.client("ssm")
    resp = ssm.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Comment=comment,
        Parameters={"commands": commands},
        TimeoutSeconds=timeout,
    )
    cmd_id = resp["Command"]["CommandId"]
    print(f"∙ SSM command {cmd_id} sent, waiting…")
    waiter = ssm.get_waiter("command_executed")
    waiter.wait(CommandId=cmd_id, InstanceId=instance_id)
    out = ssm.get_command_invocation(CommandId=cmd_id, InstanceId=instance_id)
    print(out["StandardOutputContent"])
    if out["StandardErrorContent"]:
        print("[red]STDERR[/]")
        print(out["StandardErrorContent"])
    if out["Status"] != "Success":
        raise RuntimeError(f"SSM command failed: {out['Status']}")
    return out


def check_instance_uptime(instance_id: str, region: Optional[str] = None):
    """
    Calculate and return the uptime of an EC2 instance.
    Returns a dictionary with uptime information.
    """
    sess = _session(region)
    ec2 = sess.resource("ec2")
    
    try:
        instance = ec2.Instance(instance_id)
        instance.reload()
        
        if instance.state["Name"] not in ["running", "pending"]:
            return {
                "instance_id": instance_id,
                "state": instance.state["Name"],
                "uptime": None,
                "launch_time": None,
                "error": f"Instance is {instance.state['Name']}, not running"
            }
        
        launch_time = instance.launch_time
        current_time = datetime.now(timezone.utc)
        uptime_delta = current_time - launch_time
        
        # Calculate human-readable uptime
        total_seconds = int(uptime_delta.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        uptime_str = f"{hours}h {minutes}m {seconds}s"
        if hours == 0:
            uptime_str = f"{minutes}m {seconds}s"
        if hours == 0 and minutes == 0:
            uptime_str = f"{seconds}s"
        
        return {
            "instance_id": instance_id,
            "state": instance.state["Name"],
            "instance_type": instance.instance_type,
            "launch_time": launch_time,
            "uptime_seconds": total_seconds,
            "uptime_hours": round(total_seconds / 3600, 2),
            "uptime_readable": uptime_str,
            "availability_zone": instance.placement["AvailabilityZone"]
        }
        
    except Exception as e:
        return {
            "instance_id": instance_id,
            "error": f"Failed to get instance info: {str(e)}"
        }


def run_command(instance_id: str, cmd: List[str], region: Optional[str] = None):
    sess = _session(region)
    return _ssm_send(sess, instance_id, [" ".join(cmd)], comment="ec2-helper run")


def poll_until_command_succeeds(
    instance_id: str,
    command: str,
    timeout: int = 300,
    interval: int = 10,
    region: Optional[str] = None,
):
    """
    Keep running command until it exits with code 0.
    
    Perfect for:
    - poll_until_command_succeeds(id, "systemctl status nginx")
    - poll_until_command_succeeds(id, "curl -f localhost:8000/health")
    - poll_until_command_succeeds(id, "test -f /tmp/results/*.json")
    """
    sess = _session(region)
    start_time = time.time()
    attempt = 1
    
    while time.time() - start_time < timeout:
        try:
            elapsed = int(time.time() - start_time)
            print(f"∙ Polling... (check {attempt}, {elapsed}s elapsed) ⏳")
            
            # Try to run the command
            result = _ssm_send(sess, instance_id, [command], comment="ec2-helper poll")
            
            # If we get here without exception, command succeeded
            print(f"✅ Completed after {elapsed}s")
            return result
            
        except botocore.exceptions.WaiterError as e:
            elapsed = int(time.time() - start_time)
            remaining = timeout - elapsed
            
            if remaining <= 0:
                print(f"❌ Failed after {timeout}s timeout")
                raise TimeoutError(f"Command '{command}' did not succeed within {timeout}s")
            
            time.sleep(interval)
            attempt += 1
            
        except Exception as e:
            elapsed = int(time.time() - start_time)
            remaining = timeout - elapsed
            
            if remaining <= 0:
                print(f"❌ Failed after {timeout}s timeout")
                raise TimeoutError(f"Command '{command}' did not succeed within {timeout}s")
            
            time.sleep(interval)
            attempt += 1
    
    # Timeout reached
    print(f"❌ Failed after {timeout}s timeout")
    raise TimeoutError(f"Command '{command}' did not succeed within {timeout}s")


def launch_background_process(
    instance_id: str,
    command: str,
    log_file: str = "/tmp/process.log",
    region: Optional[str] = None,
) -> str:
    """
    Launch command in background, return process info.
    
    Returns the PID of the background process for later polling.
    """
    sess = _session(region)
    
    # Wrap command to run in background and capture PID
    bg_command = f"nohup {command} > {log_file} 2>&1 & echo $!"
    
    print(f"∙ Launching background process: {command}")
    print(f"∙ Logs: {log_file}")
    
    result = _ssm_send(sess, instance_id, [bg_command], comment="ec2-helper background")
    pid = result["StandardOutputContent"].strip()
    
    print(f"∙ Background process started with PID: {pid}")
    return pid


def poll_background_process(
    instance_id: str,
    pid: str,
    success_condition: str = "test -f /tmp/done",
    timeout: int = 1800,
    interval: int = 30,
    region: Optional[str] = None,
    max_uptime_hours: Optional[float] = None,
):
    """
    Poll background process until completion.
    
    Monitors both process status and success condition.
    """
    sess = _session(region)
    start_time = time.time()
    attempt = 1
    
    while time.time() - start_time < timeout:
        elapsed = int(time.time() - start_time)
        print(f"∙ Polling... (check {attempt}, {elapsed}s elapsed) ⏳")
        
        # Check instance uptime if limit specified
        if max_uptime_hours is not None:
            uptime_info = check_instance_uptime(instance_id, region)
            if "error" not in uptime_info and uptime_info["uptime_hours"] > max_uptime_hours:
                print(f"❌ Instance has been running {uptime_info['uptime_readable']} (>{max_uptime_hours}h limit)")
                raise TimeoutError(f"Instance uptime {uptime_info['uptime_hours']:.1f}h exceeds {max_uptime_hours}h limit")
        
        try:
            # Check if success condition is met
            _ssm_send(sess, instance_id, [success_condition], comment="ec2-helper poll-bg")
            print(f"✅ Completed after {elapsed}s")
            return True
            
        except Exception:
            # Success condition not met, check if process is still running
            try:
                result = _ssm_send(sess, instance_id, [f"ps -p {pid}"], comment="ec2-helper poll-bg")
                if pid not in result["StandardOutputContent"]:
                    # Process died but success condition not met
                    print(f"❌ Process {pid} exited without success")
                    return False
                
            except Exception:
                # Process check failed, assume it's done
                print(f"❌ Unable to check process status")
                return False
        
        time.sleep(interval)
        attempt += 1
    
    print(f"❌ Failed after {timeout}s timeout")
    raise TimeoutError(f"Background process monitoring timed out after {timeout}s")


def setup_instance(
    instance_id: str,
    volume_size: int = 32,
    region: Optional[str] = None,
):
    """
    Install uv (Python package manager) and resize EBS volume on the EC2 instance.
    Default volume size is 32GB.
    """
    sess = _session(region)
    ec2 = sess.client("ec2")
    
    # 1) Resize the EBS volume
    print(f"∙ Resizing EBS volume to {volume_size}GB...")
    
    # Get the instance's root volume
    instance = sess.resource("ec2").Instance(instance_id)
    root_device = instance.root_device_name
    
    # Find the volume attached to the root device
    volumes = [v for v in instance.volumes.all() 
               if v.attachments and v.attachments[0]["Device"] == root_device]
    
    if not volumes:
        raise RuntimeError(f"Could not find root volume for instance {instance_id}")
    
    volume = volumes[0]
    current_size = volume.size
    
    if volume_size <= current_size:
        print(f"∙ Volume is already {current_size}GB (requested {volume_size}GB), skipping resize")
    else:
        # Modify the volume size
        ec2.modify_volume(VolumeId=volume.id, Size=volume_size)
        print(f"∙ Volume {volume.id} resize initiated ({current_size}GB → {volume_size}GB)")
        
        # Wait for the modification to complete
        print("∙ Waiting for volume modification to complete...")
        waiter = ec2.get_waiter("volume_in_use")
        waiter.wait(VolumeIds=[volume.id])
        
        # 2) Extend the filesystem inside the instance
        print("∙ Extending filesystem...")
        filesystem_commands = [
            # Install growpart if not available
            "sudo yum install -y cloud-utils-growpart",
            # Detect the partition and extend it
            "sudo growpart /dev/xvda 1 2>/dev/null || sudo growpart /dev/nvme0n1 1 2>/dev/null || echo 'Partition already at maximum size'",
            # Resize the filesystem (XFS for Amazon Linux 2023)
            "sudo xfs_growfs /",
            "df -h /",  # Show the new size
        ]
        _ssm_send(sess, instance_id, filesystem_commands, comment="ec2-helper volume resize")
    
    # 3) Install uv
    print("∙ Installing uv...")
    commands = [
        "sudo yum -y update -q",
        # Install UV using the official installer with proper environment
        "cd /home/ec2-user",
        "export HOME=/home/ec2-user",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        # Add UV to PATH by updating .bashrc
        "echo 'export PATH=\"/home/ec2-user/.local/bin:$PATH\"' >> /home/ec2-user/.bashrc",
        # Also create a symlink to make it globally available
        "sudo ln -sf /home/ec2-user/.local/bin/uv /usr/local/bin/uv",
        # Verify installation
        "ls -la /home/ec2-user/.local/bin/uv || echo 'UV not found in expected location'",
        "/home/ec2-user/.local/bin/uv --version || echo 'UV version check failed'",
        "echo 'uv installation completed'",
    ]
    _ssm_send(sess, instance_id, commands, comment="ec2-helper setup")


def upload_file(
    instance_id: str,
    local_path: pathlib.Path,
    remote_path: str = None,
    region: Optional[str] = None,
):
    """
    Upload a file from local machine to EC2 instance via S3.
    """
    sess = _session(region)
    s3 = sess.client("s3")
    
    # Use the same bucket pattern as setup_instance
    bucket = f"ec2-helper-{sess.client('sts').get_caller_identity()['Account']}"
    try:
        s3.head_bucket(Bucket=bucket)
    except botocore.exceptions.ClientError:
        if sess.region_name == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": sess.region_name}
            )
    
    # Generate unique S3 key for this transfer
    import uuid
    transfer_id = str(uuid.uuid4())[:8]
    s3_key = f"transfers/{instance_id}/{transfer_id}/{local_path.name}"
    
    print(f"∙ Uploading {local_path.name} to S3...")
    s3.upload_file(str(local_path), bucket, s3_key)
    
    # Generate presigned URL for download
    download_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=3600,
    )
    
    # Set remote path if not specified
    if remote_path is None:
        remote_path = f"/home/ec2-user/{local_path.name}"
    
    # Download file on EC2 instance
    commands = [
        f"curl -L '{download_url}' -o '{remote_path}'",
        f"echo 'File uploaded to {remote_path}'",
    ]
    
    print(f"∙ Downloading to instance at {remote_path}...")
    _ssm_send(sess, instance_id, commands, comment="ec2-helper upload")
    
    # Cleanup S3 object
    s3.delete_object(Bucket=bucket, Key=s3_key)
    print(f"∙ Upload complete: {local_path} → {remote_path}")


def download_file(
    instance_id: str,
    remote_path: str,
    local_path: pathlib.Path = None,
    region: Optional[str] = None,
):
    """
    Download a file from EC2 instance to local machine via S3.
    """
    sess = _session(region)
    s3 = sess.client("s3")
    
    # Use the same bucket pattern
    bucket = f"ec2-helper-{sess.client('sts').get_caller_identity()['Account']}"
    try:
        s3.head_bucket(Bucket=bucket)
    except botocore.exceptions.ClientError:
        if sess.region_name == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": sess.region_name}
            )
    
    # Generate unique S3 key for this transfer
    import uuid
    transfer_id = str(uuid.uuid4())[:8]
    filename = pathlib.Path(remote_path).name
    s3_key = f"transfers/{instance_id}/{transfer_id}/{filename}"
    
    # Generate presigned URL for upload
    upload_url = s3.generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=3600,
    )
    
    # Upload file from EC2 instance to S3
    commands = [
        f"curl -X PUT -T '{remote_path}' '{upload_url}'",
        f"echo 'File uploaded from {remote_path}'",
    ]
    
    print(f"∙ Uploading {remote_path} from instance to S3...")
    _ssm_send(sess, instance_id, commands, comment="ec2-helper download")
    
    # Set local path if not specified
    if local_path is None:
        local_path = pathlib.Path(filename)
    
    # Download file to local machine
    print(f"∙ Downloading to local path {local_path}...")
    s3.download_file(bucket, s3_key, str(local_path))
    
    # Cleanup S3 object
    s3.delete_object(Bucket=bucket, Key=s3_key)
    print(f"∙ Download complete: {remote_path} → {local_path}")


# -----------------------------------------------------
#  Typer CLI
# -----------------------------------------------------
app = typer.Typer(add_completion=False, rich_markup_mode="rich")


@app.command("spin-up")
def cli_spin_up(
    instance_type: str = typer.Option(..., help="e.g. g4dn.xlarge"),
    tag: str = typer.Option(..., help="Unique tag value to identify the box"),
    gpu: bool = typer.Option(False, help="Use GPU-optimized AMI with NVIDIA drivers"),
    dlami: bool = typer.Option(False, help="Use Deep Learning AMI with PyTorch (overrides --gpu)"),
    auto_terminate_hours: float = typer.Option(3.0, help="Hours until auto-termination during polling (default: 3.0)"),
    region: str = typer.Option(None, help="AWS region (default us-east-1)"),
):
    """Create a new EC2 instance."""
    instance, was_created = spin_up_instance(instance_type, tag, region, key_name=None, gpu=gpu, dlami=dlami, auto_terminate_hours=auto_terminate_hours)
    print(f"∙ Instance created: {was_created}")


@app.command("find")
def cli_find(
    tag: str = typer.Option(..., help="Tag value used when you spun it up"),
    region: str = typer.Option(None, help="AWS region"),
):
    """Print the instance ID if found."""
    inst = _find_instance(_session(region), tag)
    if inst:
        print(f"[bold cyan]{inst.id}[/] ({inst.instance_type} @ {inst.public_dns_name})")
    else:
        print("[red]No matching instance[/]")


@app.command("spin-up-or-find")
def cli_spinup_or_find(
    instance_type: str = typer.Option(..., help="Desired instance type"),
    tag: str = typer.Option(..., help="Tag value to search / create"),
    gpu: bool = typer.Option(False, help="Use GPU-optimized AMI with NVIDIA drivers"),
    dlami: bool = typer.Option(False, help="Use Deep Learning AMI with PyTorch (overrides --gpu)"),
    auto_terminate_hours: float = typer.Option(3.0, help="Hours until auto-termination during polling (default: 3.0)"),
    region: str = typer.Option(None, help="AWS region"),
):
    """Reuse a running instance or create one."""
    instance, was_created = spin_up_or_find(instance_type, tag, region, gpu=gpu, dlami=dlami, auto_terminate_hours=auto_terminate_hours)
    print(f"∙ Instance created: {was_created}")


@app.command("setup")
def cli_setup(
    instance_id: str = typer.Option(..., help="Instance to configure"),
    volume_size: int = typer.Option(32, help="EBS volume size in GB (default: 32)"),
    region: str = typer.Option(None, help="AWS region"),
):
    """Install uv (Python package manager) and resize EBS volume on the instance."""
    setup_instance(instance_id, volume_size, region)


@app.command("run")
def cli_run(
    instance_id: str = typer.Option(..., help="Target instance"),
    region: str = typer.Option(None, help="AWS region"),
    bash: List[str] = typer.Argument(..., help='Command to run, e.g. -- bash "ls -la"'),
):
    """Run a bash command inside the instance via SSM."""
    run_command(instance_id, bash, region)


@app.command("upload")
def cli_upload(
    instance_id: str = typer.Option(..., help="Target instance"),
    local_file: pathlib.Path = typer.Option(..., exists=True, readable=True, help="Local file to upload"),
    remote_path: str = typer.Option(None, help="Remote path (default: /home/ec2-user/filename)"),
    region: str = typer.Option(None, help="AWS region"),
):
    """Upload a file to the EC2 instance via S3."""
    upload_file(instance_id, local_file, remote_path, region)


@app.command("download")
def cli_download(
    instance_id: str = typer.Option(..., help="Target instance"),
    remote_path: str = typer.Option(..., help="Remote file path to download"),
    local_file: pathlib.Path = typer.Option(None, help="Local destination (default: filename)"),
    region: str = typer.Option(None, help="AWS region"),
):
    """Download a file from the EC2 instance via S3."""
    download_file(instance_id, remote_path, local_file, region)


@app.command("poll")
def cli_poll(
    instance_id: str = typer.Option(..., help="Target instance"),
    command: str = typer.Option(..., help="Command to poll until success"),
    timeout: int = typer.Option(300, help="Timeout in seconds (default: 300)"),
    interval: int = typer.Option(10, help="Polling interval in seconds (default: 10)"),
    region: str = typer.Option(None, help="AWS region"),
):
    """Poll command until it succeeds (exits with code 0)."""
    poll_until_command_succeeds(instance_id, command, timeout, interval, region)


@app.command("launch-bg")
def cli_launch_background(
    instance_id: str = typer.Option(..., help="Target instance"),
    command: str = typer.Option(..., help="Command to run in background"),
    log_file: str = typer.Option("/tmp/process.log", help="Log file path"),
    region: str = typer.Option(None, help="AWS region"),
):
    """Launch a command in background and return PID."""
    pid = launch_background_process(instance_id, command, log_file, region)
    print(f"Background process PID: {pid}")


@app.command("poll-bg")
def cli_poll_background(
    instance_id: str = typer.Option(..., help="Target instance"),
    pid: str = typer.Option(..., help="Process ID to monitor"),
    success_condition: str = typer.Option("test -f /tmp/done", help="Success condition command"),
    timeout: int = typer.Option(1800, help="Timeout in seconds (default: 1800)"),
    interval: int = typer.Option(30, help="Polling interval in seconds (default: 30)"),
    max_uptime_hours: float = typer.Option(None, help="Auto-terminate if instance uptime exceeds this (hours)"),
    region: str = typer.Option(None, help="AWS region"),
):
    """Poll background process until completion with optional auto-termination."""
    result = poll_background_process(instance_id, pid, success_condition, timeout, interval, region, max_uptime_hours)
    if result:
        print("✅ Background process completed successfully")
    else:
        print("❌ Background process failed")


@app.command("uptime")
def cli_uptime(
    instance_id: str = typer.Option(..., help="Instance ID to check uptime for"),
    region: str = typer.Option(None, help="AWS region"),
):
    """Check how long an instance has been running."""
    uptime_info = check_instance_uptime(instance_id, region)
    
    if "error" in uptime_info:
        print(f"[red]Error:[/] {uptime_info['error']}")
        return
    
    print(f"[bold cyan]Instance {instance_id}[/]")
    print(f"  State: {uptime_info['state']}")
    print(f"  Type: {uptime_info['instance_type']}")
    print(f"  Zone: {uptime_info['availability_zone']}")
    print(f"  Launch time: {uptime_info['launch_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  [bold green]Uptime: {uptime_info['uptime_readable']}[/] ({uptime_info['uptime_hours']} hours)")


@app.command("terminate")
def cli_terminate(instance_id: str, region: str = typer.Option(None)):
    """Terminate the instance immediately."""
    sess = _session(region)
    print(f"[red]Terminating[/] {instance_id}")
    sess.client("ec2").terminate_instances(InstanceIds=[instance_id])


def _main():
    if len(sys.argv) > 1 and sys.argv[1] == "_test":
        # quick local smoke test
        instance, was_created = spin_up_or_find("t3.micro", "smoketest", gpu=False)
        print(instance.id)
    else:
        app()


if __name__ == "__main__":
    _main()
