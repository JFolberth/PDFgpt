#!/bin/bash
nginx -t &&
service nginx start &&
streamlit run app/app.py