invert()
{ op=tr '[A-Z][a-z]''[a-z][A-Z]';
    return $op;
 }

str="cAsE"
invert $str
