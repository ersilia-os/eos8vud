FROM bentoml/model-server:0.11.0-py310
MAINTAINER ersilia

RUN pip install fpsim2==0.5.1
RUN pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install torch-geometric==2.5.3
RUN pip install torch-scatter==2.1.2
RUN pip install pandas==2.2.2
RUN pip install rdkit==2023.09.06
RUN pip install networkx==3.3
RUN pip install numpy==1.23.1
RUN pip install cdpkit==1.1.1

WORKDIR /repo
COPY . /repo