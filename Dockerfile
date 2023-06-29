
#FROM python:3.10-slim
#COPY . app
#RUN ls
#WORKDIR /app
#RUN pip install -r requirements.txt
#EXPOSE 80
#EXPOSE 8501
#RUN mkdir ~/.streamlit
#RUN cp config.toml ~/.streamlit/config.toml
#ENTRYPOINT ["streamlit", "run"]
#CMD ["app.py"]

FROM mambaorg/micromamba:0.15.3
USER root
RUN apt-get update && DEBIAN_FRONTEND=“noninteractive” apt-get install -y --no-install-recommends \
        nginx \
        ca-certificates \
        apache2-utils \
        certbot \
        python3-certbot-nginx \
        sudo \
        cifs-utils \
        && \
     rm -rf /var/lib/apt/lists/*
     
RUN apt-get update && apt-get -y install cron
RUN mkdir /opt/demo_azure
RUN chmod -R 777 /opt/demo_azure
WORKDIR /opt/demo_azure
USER micromamba
COPY environment.yml environment.yml
RUN micromamba install -y -n base -f environment.yml && \
    micromamba clean --all --yes
COPY run.sh run.sh
COPY app app
COPY nginx.conf /etc/nginx/nginx.conf
USER root
RUN chmod a+x run.sh
SHELL ["./run.sh"]