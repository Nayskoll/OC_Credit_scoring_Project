mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"davidbellaiche24@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
<<<<<<< HEAD


=======
>>>>>>> a6099779b2608e7494cef6af1335a64286d60b39
