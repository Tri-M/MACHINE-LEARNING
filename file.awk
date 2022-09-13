for f in `find . -type f `;
do 
    awk 'BEGIN { RS="." } END { print output.txt, NR - 1 }' $f 
done