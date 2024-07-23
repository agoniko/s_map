# Use Ubuntu 20.04 as base image
FROM nvidia/cuda:12.4.1-base-ubuntu20.04

# Set up environment variables
# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Install necessary packages
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    tmux \
    nano \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update
RUN apt-get install python3.10 -y
RUN apt-get install python3-pip -y


# Install ROS Noetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    ros-noetic-navigation \
    ros-noetic-octomap \
    ros-noetic-octomap-mapping

# Install catkin tools
RUN apt-get install -y \
    python3-catkin-tools

# Install PyTorch
RUN pip3 install torch torchvision torchaudio
# installing other dependencies
RUN pip3 install scikit-learn ultralytics supervision opencv-python

RUN pip3 install --upgrade pip && pip install --ignore-installed blinker PyYAML
RUN pip3 install open3d==0.18.0

# Set up ROS environment
RUN echo "source /opt/ros/noetic/setup.bash" >> /home/.bashrc
RUN echo "set -g mouse on" >> /home/.tmux.conf

RUN mkdir ~/catkin_ws
# Source ROS setup
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash && export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1"

# Clone and build rtabmap
RUN git clone https://github.com/introlab/rtabmap.git ~/rtabmap && \
    cd ~/rtabmap/build && \
    cmake .. && \
    make -j6 && \
    sudo make install


# Set the working directory
WORKDIR /home

# Start tmux by default
#CMD ["tmux"]
