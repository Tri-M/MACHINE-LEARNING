read choice
case $choice in
1) ls | tee /dev/tty | wc -l;;
2) find ./dir2 -type f -name "t*" ;; 
3) du -a . | sort -n -r | head -n 2;;
4) alias tclean="find . -name z* -exec rm -f {} \;";;
5) date | awk '{print "TIME: ", $4}';;
6) awk '!/[aeiou]/{ print $0 }' > output.txt file.txt;;
7) who | wc -l;;
# 8) uniq file1 > file3; uniq file2 >> file3; uniq -d file3;;
8) sort file1 | uniq file1>file3;sort file2 | uniq file2>>file3 | sort file3 | uniq -d;;
9) history 5 ;;
10) cp ./dir1/temp1 ./dir2/temp2;;
11) grep -r "hello";;
12) cat file1 >> cat file2 >> cat file3 >> file4;;
13) find / -type f -perm /g=w;;
*) echo "Invalid choice";;
esac

