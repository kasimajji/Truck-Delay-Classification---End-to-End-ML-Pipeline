ARG UBUNTU_VERSION=18.04
ARG UBUNTU_IMAGE_DIGEST=98706f0f213dbd440021993a82d2f70451a73698315370ae8615cc468ac06624

FROM ubuntu:${UBUNTU_VERSION}@sha256:${UBUNTU_IMAGE_DIGEST}

ARG MINICONDA_VERSION=4.12.0
ARG CONDA_CHECKSUM=3190da6626f86eee8abf1b2fd7a5af492994eb2667357ee4243975cdbb175d7a
ARG CONDA_PY_VERSION=38
ARG CONDA_PKG_VERSION=4.13.0
ARG PYTHON_VERSION=3.10.12
ARG PYARROW_VERSION=14.0.1
ARG MLIO_VERSION=v0.9.0

# Install python and other scikit-learn runtime dependencies
# Dependency list from http://scikit-learn.org/stable/developers/advanced_installation.html#installing-build-dependencies
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install --no-install-recommends \
        build-essential \
        curl \
        git \
        jq \
        libatlas-base-dev \
        nginx \
        openjdk-8-jdk-headless \
        unzip \
        wget \
        && \
    # MLIO build dependencies
    # Official Ubuntu APT repositories do not contain an up-to-date version of CMake required to build MLIO.
    # Kitware contains the latest version of CMake.
    wget http://es.archive.ubuntu.com/ubuntu/pool/main/libf/libffi/libffi7_3.3-4_amd64.deb && \
    dpkg -i libffi7_3.3-4_amd64.deb && \
    apt-get -y install --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        gnupg \
        software-properties-common \
        && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
        gpg --dearmor - | \
        tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    rm /usr/share/keyrings/kitware-archive-keyring.gpg && \
    apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        build-essential \
        cmake=3.18.4-0kitware1 \
        cmake-data=3.18.4-0kitware1 \
        doxygen \
        kitware-archive-keyring \
        libcurl4-openssl-dev \
        libssl-dev \
        libtool \
        ninja-build \
        python3-dev \
        python3-distutils \
        python3-pip \
        zlib1g-dev \
        && \
    rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    curl -L --output /tmp/Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-py${CONDA_PY_VERSION}_${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "${CONDA_CHECKSUM} /tmp/Miniconda3.sh" | sha256sum -c - && \
    bash /tmp/Miniconda3.sh -bfp /miniconda3 && \
    rm /tmp/Miniconda3.sh && \
    # Remove this when we move to Miniconda version with conda package version 4.13.0+
    rm -rf /miniconda3/pkgs/conda-4.12.0-py38h06a4308_0/info/test/*

ENV PATH=/miniconda3/bin:${PATH}
ENV PIP_ROOT_USER_ACTION=ignore

# Install MLIO with Apache Arrow integration
# We could install mlio-py from conda, but it comes  with extra support such as image reader that increases image size
# which increases training time. We build from source to minimize the image size.
RUN echo "conda ${CONDA_PKG_VERSION}" >> /miniconda3/conda-meta/pinned && \
    # Conda configuration see https://conda.io/projects/conda/en/latest/configuration.html
    conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    echo "python ${PYTHON_VERSION}.*" >> /miniconda3/conda-meta/pinned && \
    conda install -c conda-forge python=${PYTHON_VERSION} && \
    conda install conda=${CONDA_PKG_VERSION} && \
    conda update -y conda && \
    conda install -c conda-forge pyarrow=${PYARROW_VERSION} && \
    conda install pip --force-reinstall && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install pyOpenSSL==23.1.0 && \
    python3 -m pip install wheel && \
    cd /tmp && \
    git clone --branch ${MLIO_VERSION} https://github.com/awslabs/ml-io.git mlio && \
    cd mlio && \
    build-tools/build-dependency build/third-party all && \
    mkdir -p build/release && \
    cd build/release && \
    cmake -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH="$(pwd)/../third-party" ../.. && \
    cmake --build . && \
    cmake --build . --target install && \
    cmake -DMLIO_INCLUDE_PYTHON_EXTENSION=ON -DPYTHON_EXECUTABLE="/miniconda3/bin/python3" \
        -DMLIO_INCLUDE_ARROW_INTEGRATION=ON ../.. && \
    cmake --build . --target mlio-py && \
    cmake --build . --target mlio-arrow && \
    cd ../../src/mlio-py && \
    python3 setup.py bdist_wheel && \
    python3 -m pip install dist/*.whl && \
    cp -r /tmp/mlio/build/third-party/lib/libtbb* /usr/local/lib/ && \
    ldconfig && \
    rm -rf /tmp/mlio

# Install awscli
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -r aws awscliv2.zip

# Python won’t try to write .pyc or .pyo files on the import of source modules
# Force stdin, stdout and stderr to be totally unbuffered. Good for logging
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=UTF-8 LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install Scikit-Learn
# Scikit-learn 0.20 was the last version to support Python 2.7 and Python 3.4.
# Scikit-learn now requires Python 3.6 or newer.
RUN python3 -m pip install --no-cache -I scikit-learn==1.2.1
