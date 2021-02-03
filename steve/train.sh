echo ${1} > config.txt
nohup ./python &> /dev/null &
sleep 3
rm -f config.txt
