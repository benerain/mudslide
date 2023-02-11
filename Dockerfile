FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN sed -i s@/deb.debian.org/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get clean
RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y libpq-dev
RUN apt-get install -y gdal-bin
RUN apt-get install -y libgdal-dev
RUN apt-get install -y libopencv-dev
RUN apt-get install -y python3-pip python3-dev

RUN cd /usr/local/bin \
  && ln -s /usr/bin/python3 python

RUN mkdir -p /home/pieuser
WORKDIR /home/pieuser
COPY requirements.txt requirements.txt
COPY *.py ./
RUN pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
ENTRYPOINT ["python3", "mud_rock_flow.py"]