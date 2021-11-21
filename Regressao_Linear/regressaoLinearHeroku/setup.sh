mkdir -p ~/.streamlit/
showPyplotGlobalUse = False
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS=false\n\
\n\
"> ~/.streamlit/config.toml