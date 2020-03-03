cd ..

for project in 'common' 'robots-at-robots' 'robots-at-runes' 'drone-at-base' ; do
    cd ../$project || exit
    echo "Udating ${project}"
    poetry update
done
