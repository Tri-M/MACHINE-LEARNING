read choice
case $choice in
1) ls | tee /dev/tty | wc -l;;
2) find ./dir2 -type f -name "t*" ;; 
3) du -a . | sort -n -r | head -n 2;;
4) alias tclean="find . -name z* -exec rm -f {} \;";;
5) date | awk '{print "TIME: ", $4}';;
6) awk '!/[aeiou]/{ print $0 }' > output.txt file.txt;;
7) who | wc -l;;
8) uniq file1 > file3; uniq file2 >> file3; uniq -d file3;;
9) sort file1 | uniq file1>file3;sort file2 | uniq file2>>file3 | sort file3 | uniq -d;;
10) history 5 ;;
11) cp ./dir1/temp1 ./dir2/temp2;;
12) grep -r "hello";;
13) cat file1 >> cat file2 >> cat file3 >> file4;;
14) find / -type f -perm /g=w;;
*) echo "Invalid choice";;
esac

