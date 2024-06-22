# 參考了老師提供的資源以及GPT輔助完成
import hw5_gd as gd;
from micrograd.engine import Value

def f(p):
    [x, y,z] = p
    return (x-1)**2+(y-2)**2+(z-3)**2
    #return (x-2)**2+3*(y-0.5)**2+(z-3)**2
    # return x*x + 3*y*y + z*z - 4*x - 3*y - 5*z + 8
p = [Value(2.0), Value(1.0),Value(3.0)]
print(p)
print(gd.gradientDescendent(f, p))