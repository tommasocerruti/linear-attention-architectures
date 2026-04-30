# Assignment 1: Getting Started with SLURM and Clariden

This assignment will guide you through the fundamentals of using the Clariden cluster at CSCS, working with SLURM, and creating and running containers in this environment. By the end, you will have practised several essential commands, learned to submit and manage jobs, and built a custom container. Please follow the instructions closely since future assignments will build on this knowledge. To pass the assignment you need to complete all tasks, in particular task 7 which is the one you have to hand in to pass.

## [1/7] Set Up Your Access to Clariden

Clariden is the Swiss AI partition of the Alps supercomputer at CSCS that we will use in this course. Let's begin by setting up access.

1. Get an account. If all went well, we already created an account for you based on your ETH Zürich email address!

2. Go to [https://portal.cscs.ch](https://portal.cscs.ch/), log in with ETHZ username/password and follow the instructions to set up OTP (One-Time Password using an Authenticator App).

3. Install the SSH keys for you to log into the CSCS login node (ela.cscs.ch). Below we present two variants on how to do that.

    1. (Recommended) Use the CSCS keygen script (highly recommended). The keys will be valid 24h after which you will have to rerun the script.
        ```bash
        wget https://raw.githubusercontent.com/eth-cscs/sshservice-cli/main/cscs-keygen.sh
        chmod +x cscs-keygen.sh
        ./cscs-keygen.sh
        ssh-add -t 1d ~/.ssh/cscs-key
        ```
        The ssh-add command is used to add the generated SSH private keys to the ```ssh-agent```, a program that holds your private keys in memory and allows you to authenticate to servers without needing to enter the passphrase for the key each time you connect. ```-t 1d``` specifies the the **time limit** for how long the key will remain valid. At CSCS this time limit is 24h (i.e. 1 day or 1d).
        
    2. (Alternative) Manually install your keys as follows.

        1. Access the SSHService web application by accessing the URL, [https://sshservice.cscs.ch](https://sshservice.cscs.ch/) and follow the authentication instructions using your ETHZ email and Authenticator OTP.

        2. Get a signed key and download the public and private key to your machine

        3. Run the following commands to add the keys to your ssh client.
           ```bash
           mv cscs-key ~/.ssh/cscs-key
           mv cscs-key-cert.pub ~/.ssh/cscs-key-cert.pub
           chmod 0600 ~/.ssh/cscs-key
           ssh-add -t 1d ~/.ssh/cscs-key
           ```

4. With your ssh keys installed, use your username to log into the login node ela.cscs.ch and to forward further authentication requests to your local machine. The ```-A``` flag allows the SSH authentication agent running on your local machine to be forwarded to the remote server (ela.cscs.ch in this case). This means that any subsequent SSH connections made from the remote server should use the private keys stored in your local ssh-agent.
    ```bash
    ssh -A username@ela.cscs.ch
    ```

5. Once you connect to ela as instructed you should be able to ssh into Clariden!
    ```bash
    ssh username@clariden.cscs.ch
    ```

6. Great success! But let's configure the SSH config on your local machine to simplify this even further. 

    1. Make sure you are back on your machine. Use ```ctrl+d``` or just type ```exit``` to disconnect from Clariden and ela. 

    2. Add the following lines to your local machine’s ```~/.ssh/config```. Make sure to change ```username``` to your own (**twice**)!
    ```
    Host ela ela.cscs.ch
        Hostname ela.cscs.ch
        User username
        ForwardAgent yes
        IdentityFile ~/.ssh/cscs-key

    Host clariden clariden.cscs.ch
        Hostname clariden.cscs.ch
        User username
        ForwardAgent yes
        IdentityFile ~/.ssh/cscs-key
        ProxyJump ela
    ```

7. Your everyday login to Clariden now consists of two steps. 

    1. First, update your keys (requires login using your password and OTP; again, valid for 24h)
        ```bash
        ./cscs-keygen.sh
        ```

    2. Once you have the keys you simply run the following command to connect.
        ```bash
        ssh clariden
        ```

    3. If you were able to login but suddenly you get `Too many authentication failures` when logging into Clariden you might have some deprecated keys in your ssh-agent. The following command will remove all identities (keys) from the ssh-agent. Then try again the previous two steps.
       ```bash
       ssh-add -D
       ```

8. The preinstalled packages, like python, can be outdated and limiting. It's a good idea to work with your own miniconda environment. 

    1. Install miniconda by running the following commands. Follow the installation instructions but choose **NOT** to automatically initialise conda.
       ```bash
       cd ~
       wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
       chmod +x Miniconda3-latest-Linux-aarch64.sh
       ./Miniconda3-latest-Linux-aarch64.sh
       ```
       **Warning:** Do **not** let conda activate by default (i.e. do not run `conda init`). If conda is active by default, it will also be active on compute nodes, which can interfere with the container environment and cause hard-to-debug errors. Instead, to make conda available (without activating it) add `source ~/miniconda3/etc/profile.d/conda.sh` to your `~/.bashrc` file. After that you can manually enable and disable the conda environment using `conda activate` and `conda deactivate`.
       
       Notice that we used the `aarch64` architecture! All Clariden nodes use the ARM 64bit instruction set which means that we cannot use software that was built for `x86_64` which is likely what your personal machine is running (you can check on linux using `uname -m`).
    
    2. If you disconnect and reconnect into Clariden your terminal should **NOT** have a `(base)` prefix until you run `conda activate` after which you should be able to run `python --version`. Let's install a few packages that we'll need throughout the course.
       ```bash
       pip install --upgrade pip setuptools
       pip install tqdm matplotlib
       ```

9. You'll need to perform the following steps to avoid any issues later on.
    
    1. Create the following folder and file.
        ```bash
        mkdir -p $HOME/.config/containers
        vi $HOME/.config/containers/storage.conf
        ```

    2. Add the following text.
        ```
        [storage]
        driver = "overlay"
        runroot = "/dev/shm/$USER/runroot"
        graphroot = "/dev/shm/$USER/root"
        ```

10. (Optional) Setup Visual Studio Code

    1. Install visual studio code by following the official instructions: https://code.visualstudio.com/docs/setup/linux

    2. Install Remote Explorer

        1. File > Preferences > Extensions 

        2. Select Remote Explorer by Microsoft and install it

    3. Enable this setting to prevent disconnects (you need to connect to Clariden with vscode at least once before this setting appears)

        1. File > Preferences > Settings

        2. Search for: remote.SSH: Remote Server Listen On Socket

        3. Enable this setting by selecting the checkbox. 
    
    4. In vscode, now click on Remote Explorer and select clariden server (which it took from your ssh config). Once connected you should be able to navigate your home directory on the Clariden login node. If you keep having problems ensure your `ssh clariden` works as expected and manually delete `.vscode-server`on Clariden so vscode reinstalls the vscode server from scratch.


## [2/7] SLURM Basics

On Clariden, SLURM is a powerful workload manager used to allocate and schedule compute resources across the cluster, ensuring efficient and fair usage among users. By practising key SLURM commands, you can learn how to submit jobs, monitor their status, and manage computational tasks effectively. Gaining familiarity with SLURM will enable you to fully utilize the capabilities of the Clariden cluster for your research and computing needs.

1. Check available partitions (queues for jobs to run) and their nodes. This will show all partitions `-a` and in a long format `-l`. Can you tell how many idle nodes the `debug` partition has? 

    ```bash
    sinfo -a -l
    ```

2. The following command will only show your own jobs. You can use `watch` to repeatedly run a command at regular intervals (2s by default). Let's keep an eye on this while it runs as we proceed with the next steps.

    ```bash
    watch -n5 squeue --me
    ```


3. Let's create a new SSH session to Clariden. Let's now use the ```srun``` command to immediately execute an interactive job on an available node. This is useful for short test jobs. The account argument requires the group to which your account belongs. As an example, we use lsaie-ss26 but another one may apply to you.

    ```bash
    srun --account=lsaie-ss26 --time=1:00 -p debug --pty bash -c 'i=1; while [ $i -le 60 ]; do echo "Running on SLURM node... $i"; sleep 1; i=$((i+1)); done'
    ```

    `--time=01:00` specifies the runtime (in this case 1 minute). If your job is shorter the scheduler will give it higher priority and schedule it before other jobs that came earlier but will take longer! `--pty` starts an interactive session and `bash -c` will run the shell command directly.


4. After you submit your `srun` command, your job shows up under `squeue --me`. Can you tell through which states your job goes (the ST column)? 

5. If run correctly your job should result in an error. What is the reason for the error?

6. For the next steps it is important that you see the job and node id of your run. You can see this information either in the output of your `srun` and `squeue` command. The job id is a unique identifier for a submitted job, used to track and manage jobs (important for tickets!) and the node id identifies the physical compute node that allocated the job and ran your commands (on Clariden this is typically a name like `nid00xxxx`). You can always cancel a command using `scancel 123456` to cancel a specific job or `scancel --me` to cancel all your jobs.

7. After your run is completed you can access more details of your job id 123456. Given that output, you should be able to confirm your node id and see the technical details of your node. Can you tell how many CPUs and how much Memory (mem) it has?

    ```scontrol show job 123456```

8. You can also see specific details about a node as follows. CAn you tell how many GPUs it has (`Gres`)?

    ```scontrol show nodes nid001234```

9. Never run compute-intensive jobs on the login node! On Clariden you can get an interactive compute node for an hour to do very short jobs by allocating a machine interactively and simply starting a new bash shell. This can be useful to quickly process some data or build some containers as we will do later.

    ```srun --account=lsaie-ss26 -p debug --pty bash```

    Upon running the previous command you will get a shell on the compute node but it will automatically disconnect after it hits the time limit. This is useful to quickly process some data or build a container but you should not run your experiments like this! To run your experiments we will submit an sbatch script which we will look at later in this assignment.

## [3/7] Persistent Storage

All the files that are created as part of your task being executed on a compute node will be lost once the session ends. For this reason, the data you'd like to store has to be moved to a persistent storage. On the Clariden cluster, there are two mounted storage partitions: the slower but larger `/capstor` and the smaller but faster `/iopsstor`. `/capstore` is intended for long-term storage (per user quota is 150TB and 1M inodes/files) and `/iopsstor` for faster short-term access (3PB shared across all users). 

1. Your personal scratch partition is on `/iopsstor/scratch/cscs/$USER` so for easy access I suggest you add a symbolic link to your home directory.
    ```bash
    ln -s /iopsstor/scratch/cscs/$USER/ ~/scratch
    ```

2. Your personal storage partition is on `/capstor/scratch/cscs/$USER`. **Do not** write to capstor from compute nodes during a job. Always write into iopsstor.
    ```bash
    ln -s /capstor/scratch/cscs/$USER/ ~/store
    ```

3. You can check your usage quota by logging into ela.cscs.ch (it currently doesn't work on clariden). 
    ```bash
    ssh ela
    quota
    ```
    Can you tell what the quota limit is for `echo $USER`?


## [4/7] Working with Containers and Environment Files

On Clariden, containers powered by Enroot allow you to run your code in a consistent and reproducible environment, ensuring that dependencies, libraries, and system configurations align with your project's requirements. Enroot is a lightweight, rootless container runtime designed for HPC workloads, making it ideal for running Docker images without requiring elevated privileges.

To define and customize your containerized environment, you can create environment files (`.toml` files), which specify the container image to use, along with filesystem paths to mount inside it. This enables seamless integration with your project’s data and scripts while keeping your host system clean. By using Enroot, you eliminate the common "it-works-on-my-machine" issues, making your workflow more portable, reproducible, and scalable across different nodes in the Clariden cluster.

1. Create a simple `.toml` file. 
    
    We’ll start by creating a minimal environment file in your home directory on the Clariden login node. Inside the file, you specify which container image you’d like to use, any paths you want to mount, and a working directory inside the container. (We are using vim here. If you are not familiar I recommend you check out [a quick vim tutorial](https://www.youtube.com/watch?v=ggSyF1SVFr4) online)
    ```bash
    vi ~/my_env.toml
    ```
    And the following content:
    ```
    image = "/capstor/store/cscs/ethz/lsaie-ss26/nvidia+nemo+25.11.aarch64.sqsh"
    mounts = []
    workdir = "/workspace"

    [annotations]
    com.hooks.aws_ofi_nccl.enabled = "true"
    com.hooks.aws_ofi_nccl.variant = "cuda12"

    ```
    If you are wondering, the annotations are arguments to load the proper NCCL plugin that CSCS has prepared for us.

2. Launch an interactive session using your environment file. You'll have to use the full path when giving it as the environment argument.
    ```bash
    realpath ~/my_env.toml
    srun --account=lsaie-ss26 --environment=/users/thisisnotmyusername/my_env.toml -p debug --pty bash
    ```

3. The compute node has now loaded your container and started an interactive bash. Inside the container, try now to install some pip package or create a file.
    ```bash
    pip install Protego
    touch test.txt
    ```
    If you read the output carefully you should be able to tell why these fail. What is the reason? 

4. As you saw, by default the container is not writeable and fully isolated. Some tasks will require us to make changes to the container layer or access our data. For this reason, we will mount the two main filesystems that CSCS provides `/capstor` and `/iopsstor`. Let's update the toml file with the following mounts.
    ```
    mounts = [
        "/capstor",
        "/iopsstor",
        "/users",
    ]
    ```
    Next, let's add the `--container-writable` argument to our previous `srun` command. Rerun the interactive session and try again to install packages and write files. Notice that while `/capstor` and `/iopsstor` are mounted for persistent storage, changes to other paths like `/workspace` will be lost once the container session ends.

 5. Finally, you should know that you don't have to provide the full path to your toml file if you move your `my_env.toml` file on the login node to `~/.edf/` 
    ```bash
    mkdir -p ~/.edf/
    mv my_env.toml ~/.edf/.
    srun --account=lsaie-ss26 --container-writable --environment=my_env -p debug --pty bash
    ```

## [5/7] Build your Own Container

Creating your own container gives you full control over your computational environment, ensuring it remains consistent, reproducible, and portable across various systems, including HPC clusters like Clariden. With a custom container, you can preinstall all required dependencies, libraries, and configurations, reducing compatibility issues and simplifying deployments. It also helps prevent dependency conflicts while making it easy to share and version your environment with collaborators. Alternatively, you can leverage prebuilt containers from sources like NVIDIA GPU Cloud (NGC), which provides optimised environments for AI, machine learning, and HPC applications.

1. Set up Nvidia GPU Cloud (NGC) Access to use Nvidia Containers

    The Nvidia GPU Cloud (NGC) is a hub for GPU-optimised Nvidia software that provides containers, models, and more. In this tutorial, we’ll set up access to NGC.

    1. Navigate to https://ngc.nvidia.com/setup/api-key. You will have to create an account if you don't have one yet.

    2. Click the green button on the top right named “Generate API Key”.

    3. SSH into clariden

    4. Run the following commands to configure `enroot` with your API key. 
        ```bash
        mkdir -p $HOME/.config/enroot 
        cat > $HOME/.config/enroot/.credentials << EOF
        machine nvcr.io login \$oauthtoken password this-should-be-your-api-key
        machine authn.nvidia.com login \$oauthtoken password this-should-be-your-api-key
        EOF
        ```

    5. Download, and unzip ngc-cli for ARM64 from https://ngc.nvidia.com/setup/installers/cli and add it to your PATH variable.
        ```bash
        wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.44.0/files/ngccli_arm64.zip -O ngccli_arm64.zip && unzip ngccli_arm64.zip
        echo "export PATH=\"\$PATH:$HOME/ngc-cli\"" >> ~/.bash_profile
        source ~/.bash_profile
        ```

    6. Finally we can configure NGC by running the following command. Enter your API key when prompted. 
        ```bash
        ngc config set
        ```

    7. Let's edit our previous `my_env` toml file to use a container that is prebuilt for 64bit ARM and contains everything to run PyTorch on GPUs. You can find prebuilt containers for various exciting projects on the catalog.ngc.nvidia.com. Replace in your toml file the previous image with the following.
        ```
        image = "nvcr.io#nvidia/nemo:25.11"
        ```
    
    8. Get an interactive compute node and check if you can import torch. Since it will download the container from the web this may take a minute.
        ```bash
        python
        import torch
        torch.cuda.get_device_name()
        torch.cuda.device_count()
        ```
        ... and if you see the gpus. 
        ```bash
        nvidia-smi
        ```
        What is the total amount of GPU memory available on one GH200 GPU according to `nvidia-smi`?

2. In your home directory of the login node on Clariden, create a file `Dockerfile`
    ```
    FROM nvcr.io/nvidia/nemo:25.11

    # setup
    RUN apt-get update && apt-get install python3-pip python3-venv -y
    RUN pip install --upgrade pip setuptools==69.5.1
        
    # Install the rest of dependencies.
    RUN pip install \
        datasets \
        transformers \
        accelerate \
        wandb \
        dacite \
        pyyaml \
        numpy \ 
        packaging \
        safetensors \
        tqdm \
        sentencepiece \
        tensorboard \
        pandas \
        jupyter \
        deepspeed \
        seaborn

    # Create a work directory
    RUN mkdir -p /workspace
    ```
    The Dockerfile is a script that defines the steps to build a Docker container image. In this case, we build on top of the NVIDIA NeMo container (nvcr.io/nvidia/nemo:25.11), which comes pre-configured with GPU acceleration and optimized libraries for deep learning. The Dockerfile then installs system dependencies (python3-pip, python3-venv) and a collection of Python libraries for machine learning, data processing, and visualization.

    Beyond installing packages, a Dockerfile can also define environment variables, set up default commands, configure network settings, expose ports, and optimize the container size using multi-stage builds. For a deeper dive into Dockerfile see [Docker's official documentation](https://docs.docker.com/reference/dockerfile/).

3. We will now build the container. **DO NOT BUILD ON THE LOGIN NODE**. You may hit space or memory limits and it will make the login node less responsive for all other users. Before we build the container on a compute node you should get a compute node without an environment but which is container-writable. 

4. Once you are on the compute node, navigate to the folder with your Dockerfile. Then use the following command to create an image named `my_nemo:25.11` (this will take a while). Podman (short for Pod Manager) is a container management tool that allows users to run, manage, and build OCI (Open Container Initiative) containers. It is designed as a Docker-compatible alternative.
    ```bash
    podman build -t my_nemo:25.11 .
    ```

5. After you created your image you can see it in your local container registry.
    ```
    podman images
    ```

6. Use enroot to save the image into a squash file which allows you to easily share it with other users. A SquashFS file (.sqsh) is a compressed, read-only filesystem used to package and distribute software or system images efficiently. `enroot import` tells Enroot to "download" the container image from the registry and convert it into a format that Enroot can use. The `-o` (output) flag simply specifies that the converted container should be saved as a SquashFS (.sqsh) file which is a lightweight and compressed format ideal for HPC environments. This SquashFS file can be quite large. Make sure you write it into your scratch partition!
    ```bash
    cd /iopsstor/scratch/cscs/$USER
    enroot import -o my_nemo.sqsh podman://my_nemo:25.11
    ```

7. Now you can simply use your sqsh realpath as the image in your toml file to load it as your container. Try it out and check if your software packages are now available when you get a compute node.

## [6/7] Submitting a Batch Job with `sbatch`

So far, we’ve used `srun` to get an interactive session. For most production workloads or long-running experiments on HPC systems, you’ll submit jobs *non-interactively* with `sbatch`. This allows the scheduler to queue up your jobs, allocate resources when they become available, and run your commands without you needing to stay logged in.

1. Create a file named `my_first_sbatch.sh` with the following content.   
    ```
    #!/bin/bash
    #SBATCH --job-name=my_first_sbatch   # A name for your job. Visible in squeue.
    #SBATCH --nodes=1                    # Number of compute nodes to request.
    #SBATCH --ntasks-per-node=1          # Tasks (processes) per node
    #SBATCH --time=00:10:00              # HH:MM:SS, set a time limit for this job (here 10min)
    #SBATCH --partition=debug            # Partition to use; "debug" is usually for quick tests
    #SBATCH --mem=460000                 # Memory needed (simply set the mem of a node)
    #SBATCH --cpus-per-task=288          # CPU cores per task (simply set the number of cpus a node has)
    #SBATCH --environment=my_env         # the environment to use
    #SBATCH --output=/iopsstor/scratch/cscs/%u/my_first_sbatch.out  # log file for stdout / prints etc
    #SBATCH --error=/iopsstor/scratch/cscs/%u/my_first_sbatch.err  # log file for stderr / errors

    # Exit immediately if a command exits with a non-zero status (good practice)
    set -eo pipefail

    # Print SLURM variables so you see how your resources are allocated
    echo "Job Name: $SLURM_JOB_NAME"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Allocated Node(s): $SLURM_NODELIST"
    echo "Number of Tasks: $SLURM_NTASKS"
    echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
    echo "Current path: $(pwd)"
    echo "Current user: $(whoami)"

    # Demo: Write the current time to stdout every second for 10 seconds
    for i in $(seq 1 10); do
        echo "[$i] The current time is: $(date)"
        sleep 1
    done

    # Demo: Write an example message to stderr (the .err file)
    echo "This is a simulated error message going to stderr." 1>&2
    ```
    The arguments following `#SBATCH` could also be passed as command line arguments similar to the use of `srun`. 

2. Submit the job
    ```bash
    sbatch my_first_sbatch.sh
    ```
    The sbatch command will immediately return the job id. 

3. Watch `squeue --me` to see if your job has been executed. If so, let's take a look at the created logs.
    ```bash
    cat /iopsstor/scratch/cscs/$USER/my_first_sbatch.out
    cat /iopsstor/scratch/cscs/$USER/my_first_sbatch.err
    ```
    In the `.out` file you should see the time loop messages and the SLURM environment variables. In the `.err` file, you'll see the simulated error message. You can use `tail -f /iopsstor/scratch/cscs/$USER/my_first_sbatch.out` to keep monitoring the file for new content and continue displaying it in real-time (once the file has been created). You can also write both logs into the same file which can make debugging a bit easier.

## [7/7] Create a personal SLURM Cheat Sheet

If you got this far, you should have learned how to set up SSH access to the Clariden cluster using temporary signed SSH keys, explored SLURM’s basic commands for interactive and batch job submission, and become familiar with the cluster’s persistent storage layout. You also got hands-on experience creating and launching containerised environments with Enroot and Podman, building custom containers, and integrating them into your SLURM jobs. Altogether, these skills will allow you to run reproducible, high-performance workloads on Clariden which will be your foundation for the next assignments. 

However, remembering all commands can be difficult at the start. For this reason, your task is now to create **your own personal cheat sheet**. Summarise the commands that are most relevant to you to work effectively on the Clariden cluster in the form of a markdown file (not PDF). Do not submit a verbatim copy of another student’s cheat sheet as this will be considered plagiarism; it must be your own personalised summary of the steps you find most valuable. The deadline is visible on moodle. Late submissions need to be explained or else they could be considered a fail. 

To pass this assignment, **please submit your personal cheat sheet as a markdown file**.