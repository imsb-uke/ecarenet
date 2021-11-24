FROM tensorflow/tensorflow:2.1.0-gpu-py3-jupyter

# --- Install any needed packages specified in requirements.apt
COPY . /
RUN apt-get update 
RUN xargs apt-get install -y --no-install-recommends < /requirements.apt

# --- Install any needed packages specified in requirements.pip

#RUN pip install -r /requirements.pip --upgrade
RUN pip install -r /requirements_long.pip
RUN apt-get install ffmpeg libsm6 libxext6  -y


ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility
