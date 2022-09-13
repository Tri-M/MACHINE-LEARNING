# invert()
# { op=tr '[A-Z][a-z]''[a-z][A-Z]';
#     return $op;
#  }

# str="cAsE"
# invert $str
echo "Enter a number"
read num

sum=0

while [ $num -gt 0 ]
do
    mod=$((num % 10))    
    sum=$((sum + mod))   
    num=$((num / 10))    
done

echo $sum