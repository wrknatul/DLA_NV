printf "downloading LjSpeech...\n"
# axel -n 8 https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -q || printf "Failed to load ljspeech\n"
mkdir data
tar -xvf LJSpeech-1.1.tar.bz2 >> /dev/null
mv LJSpeech-1.1 data/LJSpeech-1.1
