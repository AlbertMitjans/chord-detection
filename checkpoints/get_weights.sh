wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1n7Cwc7UWHaZ88vGpPoLlX_yJnAu1L6TT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1n7Cwc7UWHaZ88vGpPoLlX_yJnAu1L6TT" -O ckpts.zip && rm -rf /tmp/cookies.txt

unzip ckpts.zip

rm ckpts.zip

