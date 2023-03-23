FROM nvidia/cuda:11.3.0-base-ubuntu20.04

RUN apt-get update && \
    apt-get install -y software-properties-common


RUN apt-get update && apt-get install -y wget
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update
RUN apt-get update --fix-missing
RUN apt-get dist-upgrade --yes
RUN apt-get install -y libgomp1 libitm1 libatomic1 liblsan0 libtsan0 libubsan0 libcilkrts5 libquadmath0 libmpc3 binutils libc6-dev

RUN wget http://launchpadlibrarian.net/247707088/libmpfr4_3.1.4-1_amd64.deb
RUN wget http://launchpadlibrarian.net/253728424/libasan1_4.9.3-13ubuntu2_amd64.deb
RUN wget http://launchpadlibrarian.net/253728426/libgcc-4.9-dev_4.9.3-13ubuntu2_amd64.deb
RUN wget http://launchpadlibrarian.net/253728314/gcc-4.9-base_4.9.3-13ubuntu2_amd64.deb
RUN wget http://launchpadlibrarian.net/253728399/cpp-4.9_4.9.3-13ubuntu2_amd64.deb
RUN wget http://launchpadlibrarian.net/253728404/gcc-4.9_4.9.3-13ubuntu2_amd64.deb
RUN wget http://launchpadlibrarian.net/253728432/libstdc++-4.9-dev_4.9.3-13ubuntu2_amd64.deb
RUN wget http://launchpadlibrarian.net/253728401/g++-4.9_4.9.3-13ubuntu2_amd64.deb

RUN  dpkg -i gcc-4.9-base_4.9.3-13ubuntu2_amd64.deb
RUN  dpkg -i libmpfr4_3.1.4-1_amd64.deb
RUN  dpkg -i libasan1_4.9.3-13ubuntu2_amd64.deb
RUN  dpkg -i libgcc-4.9-dev_4.9.3-13ubuntu2_amd64.deb
RUN  dpkg -i cpp-4.9_4.9.3-13ubuntu2_amd64.deb
RUN  dpkg -i gcc-4.9_4.9.3-13ubuntu2_amd64.deb
RUN  dpkg -i libstdc++-4.9-dev_4.9.3-13ubuntu2_amd64.deb
RUN  dpkg -i g++-4.9_4.9.3-13ubuntu2_amd64.deb

RUN apt-get install --only-upgrade libstdc++6


RUN apt-get update \
    && apt-get install -y wget cmake g++ atop \
    && apt-get install -y libsm6 libxext6 libxrender-dev\
    && apt-get install -y ffmpeg pkg-config \
    && apt-get install -y libglib2.0-0 libnss3 libgconf-2-4 libfontconfig1 \
    && apt-get install --reinstall ca-certificates \
    && apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libavdevice-dev \
    && apt-get install -y git libssl-dev libboost-all-dev libopenblas-dev liblapack-dev \
    && apt-get install -y libxcursor-dev libxi-dev libglu1-mesa-dev libgl1-mesa-dev libao-dev \
    && apt-get install -y python3.8 \
    && apt-get install -y python3-dev \
    && rm -rf /var/lib/apt/lists/*


RUN wget -nv https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh -O ~/anaconda.sh \
    && bash ~/anaconda.sh -b -p /opt/conda && rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
ENV PATH /opt/conda/bin:$PATH

COPY conda_env.yml /
RUN conda env create -n main -f conda_env.yml && echo "conda activate main" >> ~/.bashrc

RUN conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch -c nvidia
RUN pip install omegaconf>=2.0.0 pytorch-lightning>=1.0.8 einops transformers pandas

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
