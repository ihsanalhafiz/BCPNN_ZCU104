# BCPNN on ZCU104

This project demonstrates the deployment and execution of a BCPNN (Bayesian Confidence Propagation Neural Network) kernel on the Xilinx ZCU104 board using a Vitis acceleration flow with PAC (Platform Acceleration Component).

---

## System Setup & Preparation

### Prerequisites

- ZCU104 Evaluation Board
- Ubuntu 22.04 LTS running on the board
- Cross-compilation toolchain: `aarch64-linux-gnu-g++`

---

## 1. ZCU104 Setup with Certified Ubuntu

1. Follow the official instructions from Xilinx to install Certified Ubuntu 22.04 on ZCU104:  
   [Getting Started with Certified Ubuntu 22.04 LTS for Xilinx Devices](https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/2363129857/Getting+Started+with+Certified+Ubuntu+22.04+LTS+for+Xilinx+Devices)

2. Install the Xilinx configuration tool:

   ```bash
   sudo apt update
   sudo apt install xlnx-config
   ```

3. Verify the installation:

   ```bash
   xlnx-config -q
   ```

---

## 2. Prepare the Sysroot for Cross-Compilation

1. Download the sysroot for ZCU104:

   [Download Sysroot](https://people.canonical.com/~platform/images/xilinx/zcu-ubuntu-22.04/)

   File:  
   `iot-limerick-zcu-classic-desktop-2204-x05-2-20221123-58-sysroot.tar.xz`

2. Extract the tar file:

   ```bash
   tar -xf iot-limerick-zcu-classic-desktop-*.tar.xz
   ```

   You should now have a folder named `sysroots/`.

---

## 3. Install PAC (Platform Acceleration Component)

1. Copy the PAC container:

   ```bash
   sudo cp -r ./PAC_container /boot/firmware/xlnx-config/
   sudo cp -r ./PAC_container /usr/local/share/xlnx-config/
   ```

2. Re-check configuration status:

   ```bash
   xlnx-config -q
   ```

3. Install the PAC:

   ```bash
   sudo xlnx-config -a stream_32x128_SP
   ```

4. Reboot the system:

   ```bash
   sudo reboot now
   ```

5. After reboot, verify PAC activation:

   ```bash
   xlnx-config -q
   ```

   Make sure `stream_32x128_SP` is marked as active.

---

## 4. Compile the Host Program

### General Format

```bash
make host \
  CXX=/usr/bin/aarch64-linux-gnu-g++ \
  HOST_ARCH=aarch64 \
  EDGE_COMMON_SW=<PATH_TO_SYSROOT_PARENT_DIRECTORY> \
  HOST_COMPILE=<TARGET_CPP_FILE_WITHOUT_EXTENSION>
```

### Example

If your sysroot is in `/home/ubuntu/sysroots`, run:

```bash
make host \
  CXX=/usr/bin/aarch64-linux-gnu-g++ \
  HOST_ARCH=aarch64 \
  EDGE_COMMON_SW=/home/ubuntu \
  HOST_COMPILE=mnistmain_FPGA
```

 **Note:**  
- `HOST_COMPILE` should match the base filename of the `.cpp` file you want to compile.
- For inference-only execution, use `mnistmain_FPGA_inference.cpp` instead.

---

## 5. Run the Application

### Run Format

```bash
./<binary_file> <parameter_file> <xclbin_file> <trained_data_output_file>
```

before run, make sure you have correct dataset path from parameters file. if not, you need to run this code to get correct bin dataset
[extract MNIST dataset](https://github.com/nbrav/BCPNNSim-ReprLearn/blob/main/Data/mnist/extract.py)

or if you want to have it ready bin dataset, you can extract `Data.zip`

### Example

```bash
./mnistmain_FPGA \
  ./test/MNIST_ZCU104/mnistmain.par \
  ./PAC_container/hwconfig/stream_32x128_SP/zcu104/BCPNN_Kernel.xclbin \
  trained_data.bin
```

This command executes training and saves the result as `trained_data.bin`.

For **inference**, repeat the same structure using the inference binary.

---

## Directory Structure

```text
+-- PAC_container/
   +-- hwconfig/stream_32x128_SP/zcu104/BCPNN_Kernel.xclbin
+-- test/
   +-- MNIST_ZCU104/mnistmain.par
+-- mnistmain_FPGA.cpp
+-- mnistmain_FPGA_inference.cpp
+-- trained_data.bin
+-- Makefile
```

---

## Notes

- Ensure the `.xclbin` file matches the PAC installed on your board.
- Use the same sysroot version as your ZCU104's Ubuntu image to avoid library mismatches.
- you can change the parameter file to change size of dataset for train and test

# Authors
M Ihsan Al Hafiz (miahafiz@kth.se),
Naresh Ravichandran (nbrav@kth.se),
Anders Lansner (ala@kth.se),
Pawel Herman (paherman@kth.se),
Artur Podobas (podobas@kth.se).

# Supported by

EXTRA-BRAIN project, funded by the European Union under grant no. 101135809. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for them.
