#!/bin/bash

streamlit run --server.headless true Program.py --server.port 8581 &

sleep 5

electron http://localhost:8581/
