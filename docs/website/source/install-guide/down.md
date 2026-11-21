{#down}
# Installing NCCL



In order to download NCCL, ensure you are registered for the [NVIDIA Developer Program](https://developer.nvidia.com/accelerated-computing-developer).

1.  Go to: [NVIDIA NCCL home page](https://developer.nvidia.com/nccl).

2.  Click **Download**.

3.  Complete the short survey and click **Submit**.

4.  Accept the Terms and Conditions. A list of available download versions of NCCL displays.

5.  Select the NCCL version you want to install. A list of available resources displays. Refer to the following sections to choose the correct package depending on the Linux distribution you are using.



{#debian}
## Ubuntu



Installing NCCL on Ubuntu requires you to first add a repository to the APT system containing the NCCL packages, then installing the NCCL packages through APT. There are two repositories available; a local repository and a network repository. Choosing the latter is recommended to easily retrieve upgrades when newer versions are posted.

In the following commands, please replace `<architecture>` with your CPU architecture: `x86_64`, `ppc64le`, or `sbsa`, and replace `<distro>` with the Ubuntu version, for example `ubuntu1604`, `ubuntu1804`, or `ubuntu2004`.

1.  Install the repository.

    -   For a local NCCL repository:


        ```
        sudo dpkg -i nccl-repo-<version>.deb
        ```



        :::{note}
        The local repository installation will prompt you to install the local key it embeds and with which packages are signed. Make sure to follow the instructions to install the local key, or the install phase will fail later.

        :::

    -   For the network repository:


        ```
        wget https://developer.download.nvidia.com/compute/cuda/repos/<distro>/<architecture>/cuda-keyring_1.0-1_all.deb
        ```


        ```
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
        ```


2.  Update the APT database:


    ```
    sudo apt update
    ```

3.  Install the `libnccl2` package with APT. Additionally, if you need to compile applications with NCCL, you can install the `libnccl-dev` package as well:



    :::{note}
    If you are using the network repository, the following command will upgrade CUDA to the latest version.
    :::


    ```
    sudo apt install libnccl2 libnccl-dev
    ```

    If you prefer to keep an older version of CUDA, specify a specific version, for example:


    ```
    sudo apt install libnccl2=2.4.8-1+cuda10.0 libnccl-dev=2.4.8-1+cuda10.0
    ```

    Refer to the [download page](https://developer.nvidia.com/nccl) for exact package versions.



{#rhel_centos}
## RHEL/CentOS



Installing NCCL on RHEL or CentOS requires you to first add a repository to the YUM system containing the NCCL packages, then installing the NCCL packages through YUM. There are two repositories available; a local repository and a network repository. Choosing the latter is recommended to easily retrieve upgrades when newer versions are posted.

In the following commands, `<architecture>` should be your CPU architecture: `x86_64`, `ppc64le`, or `sbsa`.

1.  Install the repository.

    -   For a local NCCL repository:


        ```
        sudo rpm -i nccl-repo-<version>.rpm
        ```

    -   For a network repository:

        RHEL 7:


        ```
        sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/<architecture>/cuda-rhel7.repo
        ```

        RHEL 8:


        ```
        sudo dnf config-manager --add-repo http://developer.download.nvidia.com/compute/cuda/repos/rhel8/<architecture>/cuda-rhel8.repo
        ```


2.  Install the `libnccl2` package with YUM. Additionally, if you need to compile applications with NCCL, you can install the `libnccl-devel` package and optionally the `libnccl-static` package if you intend to link NCCL statically in your application:



    :::{note}
    If you are using the network repository, the following command will upgrade CUDA to the latest version.
    :::


    ```
    sudo yum install libnccl libnccl-devel libnccl-static
    ```

    If you prefer to keep an older version of CUDA, specify a specific version, for example:


    ```
    sudo yum install libnccl-2.4.8-1+cuda10.0 libnccl-devel-2.4.8-1+cuda10.0 libnccl-static-2.4.8-1+cuda10.0
    ```

    Refer to the [download page](https://developer.nvidia.com/nccl) for exact package versions.



{#tar}
## Other Distributions



Download the tar file package. For more information, see {ref}`Installing NCCL <down>`.

1.  Extract the NCCL package to your home directory or in `/usr/local` if installed as root for all users:


    ```
    # cd /usr/local 
    # tar xvf nccl-<version>.txz
    ```

2.  When compiling applications, specify the directory path to where you installed NCCL, for example `/usr/local/nccl-<version>/`.



