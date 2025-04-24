FROM python:3.9
#FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:2.6.2

WORKDIR /usr/src/app

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip --no-cache-dir --progress-bar off
# 安装 Python 依赖
#COPY requirements.txt .
#RUN pip install --upgrade pip && \
#    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制代码
COPY . .
RUN chmod +x ./entrypoint.sh
EXPOSE 5000
ENTRYPOINT ["./entrypoint.sh"]
CMD ["https://pypi.tuna.tsinghua.edu.cn/simple"]
#CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "main:app"]