FROM qwenllm/qwenvl:2.5-cu121

RUN apt update && apt install -y ffmpeg

WORKDIR /data/shared/Qwen/

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . ./qwen-vl-finetune

CMD ["/bin/bash"]