# TSBB17 Project 1: Visual Object Recognition

This toolkit provides basic code for training convolutional neural networks for object recognition on the CIFAR-10 dataset.

## Python (Pytorch)
We recommend using conda to manage Python packages and environments.
Below are instructions on how to access conda in the ISY computer rooms,
on NSC clusters, or installing on your own machine.

### Preparations
#### Running in ISY computer rooms (Olympen, etc.)
1. Load the TSBB17 module (it mounts the course directory and adds
    the path to our conda installation): **Repeat this step on each login!**
    ```
    module add courses/TSBB17
    ```
[//]: #   (cd /courses/TSBB17)
[//]: #   (export PATH="/courses/TSBB17/miniconda3/bin:$PATH")

2. Initialize conda: **Only needed once for each user**
    ```
    conda init bash
    ```
3. Start a new bash shell, and then activate the conda environment:
    ```
    conda activate tsbb17
    ```

You should now see `(tsbb17)` before the current path in the terminal which indicates that the virtual environment was correctly activated.

#### Running on your own machine
1. Install MiniConda

    ```
    $ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    $ chmod +x Miniconda3-latest-Linux-x86_64.sh 
    $ ./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/sw/miniconda3
    ```

2. Create a Conda environment named "tsbb17", with PyTorch installed

    ```
    $ conda create --name tsbb17 python=3 pytorch torchvision -c pytorch
    ```

3. Repeat steps 2,3 from above.

### Running the code
1. Download the repository
    
    ```
    $ wget https://gitlab.ida.liu.se/tsbb17/visual-object-recognition/-/archive/master/visual-object-recognition-master.zip
    $ unzip visual-object-recognition-master.zip
    $ cd visual-object-recognition-master
    ```
2. Activate the environment if it is not activated. **You need to do this everytime you work on your project**

    ```
    $ conda activate tsbb17
    ```
After the environment is activated you can use the `python3` command to run
your Python scripts.
To stop using the environment run `conda deactivate`.


### Pytorch code explanation
The startup script is **main.py** which will download the CIFAR10 dataset, initialize the cvlNet and starts the training process. When the training is finished it evaluates the trained model on the test set. The script **train.py** has the training code and **test.py** has the test code.

After each training epoch, a checkpoint file is saved to /path/to/visual-object-recognition/python/checkpoints. You can write code to load checkpoints to be able to resume training.

## Matlab (MAY WORK BUT NOT SUPPORTED ANYMORE)
### Dependencies
Matlab (preferably >= 2015b).

### Installation

The following commands will download the toolkit
```
1. Clone the repository: git clone git@gitlab.ida.liu.se:tsbb17/visual-object-recognition.git
2. cd visual-object-recognition
3. Download MatConvNet: git submodule init
                        git submodule update
```

Follow the instructions bellow to install MatConvNet:

```
1. Open the Matlab editor.
2. In the Matlab editor, navigate to /path/to/visual-object-recognition/matconvnet/matlab.
3. Run vl_compilenn in the Matlab command window to initiate the installation. In order to compile MatConvNet 
   with GPU functionality, run vl_compilenn('EnableGpu', true). This requires you to have cuda installed on 
   your computer, in which case you also might need to specify the path to the cuda folder: 
   vl_compilenn('EnableGpu', true, 'CudaRoot', '/path/to/cuda').
4. When the installation is completed move all files in the folder /path/to/visual-object-recognition/matconvnet/matlab/mex
   to the folder /path/to/visual-object-recognition/matconvnet/matlab and you are good to go!
```

There seems to be problems with running Cuda on the lab computers with the latest versios of MatConvNet. To resolve this, you can checkout an earlier version before installing using

```
cd matconvnet
git checkout 6221423ac64067426bbaa2ffd31557523258d1fe
```


### Code Explaination

The toolkit is built on top of MatConvNet specifically for training DagNN-networks on the CIFAR-10 dataset. In order to train a network you need a **network file** and a **configuration file**. The script utils/getCifarimdb.m downloads the CIFAR-10 dataset and generates an imdb.mat-file, which is used for the training. The default getCifarimdb script includes the generation of two colorspaces, rgb and gray, however you can modify the file to add different colorspaces. 
During training the network file is called to construct the network, the imdb.mat-file is loaded and then images are loaded from the imdb.mat-file by the getDagNNBatch.m which fetches a training batch from the imdb.mat-file.
*Note:* If you added a new colorspace to getCifarimdb.m, you should modify getDagNNBatch.m as well to pass it to the network. 

### Run test

The toolkit comes with an implementation of a test network called cvlNet found in */network* folder. Run the configuration file **train_cnn_cifar** in the top folder to start training the cvlNet on CIFAR-10.

### Create and train your own network

The first step is to create your network file and put it in the */network* folder. To train the network on the CIFAR-10
dataset you need to generate a configuration script. The toolkit provides a configuration script for the cvlNet 
in train_cnn_cifar.m which you can use as a template. 

In order to train your own network you should specify the name of the network in 

```
% Network options 
opts.networkName = 'yourNetworkName' ;                         % Network file name 

```
this name should match the network file in */networks*.

You also need to specify the input data colorspaces in

```
opts.colorspaces = {'rgb', 'gray'} ;
                                                             
```
*Note:* Only RGB and Gray and implemented. You should implement other colorspaces yourself.

Here you can add any input data as long as it exist in the imdb.mat-file for the specific network. If more than one colorspace is specified the configuration expects a multistream network. See the comments in train_cnn_cifar.m and utils/getCifarimdb.m for more implementation details.

Afterwards, you need to modify getDagNNBatch.m to feed different colorspaces to the network. Currently, the input variable to the network is called *input_<colorspace>*.

In case that you changed some data parameters and you want to regenerate the imdb.mat-file, just set the following variable to *true*
```
opts.regenIMDB = true ;
```

### Visualize network

Use the following script to visualize the network
```
net.print({'input_rgb',[32 32 3 1]},'format','dot','all',true);
```

