FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt update && apt upgrade -y
RUN apt install -y python3 python3-pip
RUN rm -rf /var/cache/apt/archives /var/lib/apt/lists/*.

EXPOSE 80

WORKDIR /root/code

RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install scipy
RUN pip3 install scikit-learn

RUN pip3 install torch
RUN pip3 install sentence_transformers
RUN pip3 install transformers
RUN pip3 install streamlit
RUN pip3 install altair 

COPY ./code /root/code
# CMD tail -f /dev/null
CMD streamlit run app.py --server.port=80 --server.address=0.0.0.0